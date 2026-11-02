"""Factor -> asset transmission models.

Each asset's monthly price return is explained by the same month's macro factor
changes (contemporaneous transmission), optionally with lags, an
inflation-regime interaction, extra candidate factors / structural shocks, or a
small neural network instead of a linear map. A linear spec may also give single
assets equations of their own: extra regressors (asset_factors, e.g. base metals
relative to oil for energy) or a later first month (asset_start, e.g. gold from
1982). The coefficient table stays one matrix over the union of the features,
0 where a feature is not in an asset's equation. The diagnostics compare these
variants out of sample; the scenario engine uses whichever wins.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .. import config

# Assets whose returns are regressed on factors. Treasury, crude oil and cash
# are also fitted (for transmission charts), but simulations build them
# structurally from the yield, oil and policy-rate paths.
MODEL_ASSETS = [a for a in config.ASSET_IDS if a != "cash"]   # cash has no price return
STRUCTURAL_ASSETS = ("treasury_10y", "crude_oil", "cash")
HIGH_INFLATION = 4.0  # CPI YoY (%) above which the inflation regime is "high"
# An asset's own later first month (asset_start) is used only when its fit window still holds this many
# months; shorter windows (the forecast's early origins, per-era fits, early backtest refits) fall back to
# the common start.
MIN_ASSET_WINDOW = 120
# Asset names in describe() (the diagnostics page's short labels; config.ASSETS has the long ones).
SHORT_NAMES = {"us_equity": "美股大盘", "nasdaq": "纳指100", "energy": "能源股", "industrials": "工业股",
               "staples": "必需消费股", "healthcare": "医疗保健股", "utilities": "公用事业股", "financials": "金融股",
               "real_estate": "REITs", "gold": "黄金", "crude_oil": "原油", "treasury_10y": "10年美债"}


def _pairs(value) -> list:
    """(key, value) pairs of a dict, or of a list of pairs (a spec read back from JSON); None = none."""
    if value is None:
        return []
    return list(value.items()) if isinstance(value, dict) else [tuple(p) for p in value]


def _norm_asset_factors(value) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """{asset: [factor ids]} or ((asset, ids), ...) -> ((asset, sorted unique ids), ...) sorted by asset,
    assets without extras dropped, so equal specs compare (and hash) equal."""
    out: dict[str, set] = {}
    for asset, ids in _pairs(value):
        ids = (ids,) if isinstance(ids, str) else (ids or ())
        out.setdefault(str(asset), set()).update(str(f) for f in ids)
    return tuple((a, tuple(sorted(ids))) for a, ids in sorted(out.items()) if ids)


def _norm_asset_start(value) -> tuple[tuple[str, str], ...]:
    """{asset: "YYYY-MM"} or ((asset, month), ...) -> ((asset, "YYYY-MM"), ...) sorted by asset."""
    out = {str(a): pd.Timestamp(str(m)).strftime("%Y-%m") for a, m in _pairs(value) if m is not None}
    return tuple(sorted(out.items()))


@dataclass(frozen=True)
class ModelSpec:
    factors: tuple[str, ...] = tuple(config.CORE_FACTOR_IDS)
    lags: int = 0
    regime: str | None = None          # None or "inflation"
    kind: str = "linear"               # "linear" or "mlp"
    ridge: float = 0.0                 # L2 penalty on standardized coefficients
    window_years: int | None = None    # None = all training data
    target: str = "p"                  # "p" price return, "r" total return
    shock_cols: tuple[str, ...] = ()   # extra panel columns, e.g. structural shocks
    hidden: int = 16                   # MLP hidden units
    asset_factors: tuple[tuple[str, tuple[str, ...]], ...] = ()   # ((asset, (factor ids)), ...): extra regressors
    asset_start: tuple[tuple[str, str], ...] = ()                 # ((asset, "YYYY-MM"), ...): that equation's fit start
    name: str = "base"

    def __post_init__(self):
        # dicts or lists (from_dict, a hand-written spec) become sorted tuples: hashable, order-free
        object.__setattr__(self, "asset_factors", _norm_asset_factors(self.asset_factors))
        object.__setattr__(self, "asset_start", _norm_asset_start(self.asset_start))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["factors"] = list(self.factors)
        d["shock_cols"] = list(self.shock_cols)
        d["asset_factors"] = {a: list(ids) for a, ids in self.asset_factors}
        d["asset_start"] = dict(self.asset_start)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        """Inverse of to_dict; asset_factors / asset_start may be dicts or lists of pairs, or absent (a
        model_selection.json written before they existed loads unchanged)."""
        d = dict(d)
        d["factors"] = tuple(d.get("factors", config.CORE_FACTOR_IDS))
        d["shock_cols"] = tuple(d.get("shock_cols", ()))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def describe(self) -> str:
        parts = [f"{len(self.factors)} 个因子"]
        if self.shock_cols:
            parts.append(f"+{len(self.shock_cols)} 个结构冲击")
        if self.lags:
            parts.append(f"{self.lags} 阶滞后")
        if self.regime:
            parts.append("通胀状态交互")
        if self.kind == "mlp":
            parts.append(f"小型神经网络({self.hidden})")
        if self.ridge:
            parts.append(f"岭回归 λ={self.ridge}")
        if self.window_years:
            parts.append(f"最近 {self.window_years} 年窗口")
        extras = asset_extras(self)
        if len(extras) == 1:
            (a, ids), = extras.items()
            parts.append(f"{SHORT_NAMES.get(a, a)} +{len(ids)} 个专门因子")
        elif extras:
            parts.append(f"{len(extras)} 个资产有专门因子")
        for a, month in self.asset_start:
            year, mon = month.split("-")
            parts.append(f"{SHORT_NAMES.get(a, a)}从 {year} 年{'' if mon == '01' else f' {int(mon)} 月'}起估计")
        return "、".join(parts)


def asset_extras(spec: ModelSpec) -> dict[str, tuple[str, ...]]:
    """Each asset's extra factor ids, leaving out those that are common factors already."""
    common = set(spec.factors)
    out = {}
    for a, ids in spec.asset_factors:
        own = tuple(f for f in ids if f not in common)
        if own:
            out[a] = own
    return out


def _extra_ids(spec: ModelSpec) -> list[str]:
    """The union of the assets' extras: sorted, each once."""
    return sorted({f for ids in asset_extras(spec).values() for f in ids})


def _common_columns(spec: ModelSpec) -> list[str]:
    return [f"f_{f}" for f in spec.factors] + list(spec.shock_cols)


def base_columns(spec: ModelSpec) -> list[str]:
    """Panel columns of the model: the common factors, the shocks, then the union of the assets' extras."""
    return _common_columns(spec) + [f"f_{f}" for f in _extra_ids(spec)]


def asset_columns(spec: ModelSpec, asset: str) -> list[str]:
    """Panel columns of one asset's equation: the common ones, then its own extras (base_columns order)."""
    own = set(asset_extras(spec).get(asset, ()))
    return _common_columns(spec) + [f"f_{f}" for f in _extra_ids(spec) if f in own]


def design_matrix(panel: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Feature matrix for every month of `panel` (NaN where lags are missing).

    Order: the common columns, their lags, their high-inflation terms and the dummy, then the extras
    (contemporaneous only: lags and the regime interaction apply to the common columns)."""
    base = panel[_common_columns(spec)].copy()
    base.columns = [c[2:] if c.startswith("f_") else c for c in base.columns]
    parts = [base]
    for lag in range(1, spec.lags + 1):
        parts.append(base.shift(lag).add_suffix(f"_l{lag}"))
    if spec.regime == "inflation":
        dummy = (panel["s_cpi_yoy"].shift(1) > HIGH_INFLATION).astype(float)
        inter = base.mul(dummy, axis=0).add_suffix("_hi")
        parts.append(inter)
        parts.append(dummy.rename("hi_infl").to_frame())
    extra = _extra_ids(spec)
    if extra:
        parts.append(panel[[f"f_{f}" for f in extra]].set_axis(extra, axis=1))
    return pd.concat(parts, axis=1)


def feature_names(spec: ModelSpec) -> list[str]:
    names = [c[2:] if c.startswith("f_") else c for c in _common_columns(spec)]
    out = list(names)
    for lag in range(1, spec.lags + 1):
        out += [f"{n}_l{lag}" for n in names]
    if spec.regime == "inflation":
        out += [f"{n}_hi" for n in names] + ["hi_infl"]
    return out + _extra_ids(spec)


def asset_feature_names(spec: ModelSpec, asset: str) -> list[str]:
    """The features of one asset's equation, in feature_names order (its coefficients on the others are 0)."""
    other = set(_extra_ids(spec)) - set(asset_extras(spec).get(asset, ()))
    return [f for f in feature_names(spec) if f not in other]


def features_from_paths(spec: ModelSpec, base_paths: np.ndarray, base_init: np.ndarray,
                        hi_infl: np.ndarray | None) -> np.ndarray:
    """Build the feature tensor for simulated paths (feature_names order).

    base_paths: (n_paths, T, n_base) simulated changes of the base_columns, in
                that order: the common columns, then the extras.
    base_init:  (lags, n_base) the last actual months before the simulation
                start, oldest first, used to fill lags at the start of a path.
    hi_infl:    (n_paths, T) high-inflation dummy (needed if spec.regime).
    Lags and the regime terms are built from the common columns; the extras
    are appended as they are.
    """
    n, T, k = base_paths.shape
    c = len(_common_columns(spec))
    common = base_paths if c == k else base_paths[:, :, :c]
    parts = [common]
    if spec.lags:
        init = np.broadcast_to(base_init[None, :, :c], (n, spec.lags, c))
        ext = np.concatenate([init, common], axis=1)              # (n, lags+T, c)
        for lag in range(1, spec.lags + 1):
            parts.append(ext[:, spec.lags - lag: spec.lags - lag + T, :])
    if spec.regime == "inflation":
        d = hi_infl[:, :, None]
        parts.append(common * d)
        parts.append(d)
    if c < k:
        parts.append(base_paths[:, :, c:])
    return np.concatenate(parts, axis=2)


@dataclass
class FitStats:
    r2: dict = field(default_factory=dict)
    n_obs: dict = field(default_factory=dict)
    resid_sd: dict = field(default_factory=dict)


def _linear(z: np.ndarray, y: np.ndarray, x_mean: np.ndarray, x_sd: np.ndarray,
            ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS / ridge of y on standardized features z: (coef (k, assets) in original units, intercept,
    t statistics (k, assets))."""
    n, k = z.shape
    ym = y.mean(axis=0)
    A = z.T @ z + ridge * n * np.eye(k)
    beta_z = np.linalg.solve(A, z.T @ (y - ym))           # (k, assets)
    coef = beta_z / x_sd[:, None]
    intercept = ym - x_mean @ coef
    # OLS-style t statistics (exact only when ridge == 0)
    resid = (y - ym) - z @ beta_z
    dof = max(n - k - 1, 1)
    sigma2 = (resid ** 2).sum(axis=0) / dof
    try:
        inv = np.linalg.inv(A)
        cov_diag = np.diag(inv @ (z.T @ z) @ inv)
    except np.linalg.LinAlgError:
        cov_diag = np.full(k, np.nan)
    se = np.sqrt(np.outer(cov_diag, sigma2))              # (k, assets) in z units
    return coef, intercept, beta_z / se


class AssetFactorModel:
    """One spec, fitted per asset on a training window.

    Every equation is fitted on the same months (every feature and every model asset's return present),
    except that an asset with an asset_start starts later; assets are grouped by (extras, start) and each
    group is one OLS / ridge on its own columns and months. coef is assets x the union of the features,
    0.0 where a feature is not in an asset's equation (tstat NaN there), so predictions, contributions and
    the attribution need no special case.
    """

    def __init__(self, spec: ModelSpec | None = None, assets: list[str] | None = None):
        self.spec = spec or ModelSpec()
        self.assets = list(assets or MODEL_ASSETS)
        self.features = feature_names(self.spec)
        self.x_mean: np.ndarray | None = None
        self.x_sd: np.ndarray | None = None
        self.coef: pd.DataFrame | None = None       # assets x features (original units)
        self.intercept: pd.Series | None = None
        self.tstat: pd.DataFrame | None = None
        self.mlp = None
        self.y_mean: np.ndarray | None = None
        self.y_sd: np.ndarray | None = None
        self.stats = FitStats()
        self.train_start = None
        self.train_end = None
        self.asset_train_start: dict[str, str] = {}          # asset -> first month of its own fit ("YYYY-MM")
        self.asset_start_used: dict[str, str | None] = {}    # asset with an asset_start -> it, or None (fell back)

    # ------------------------------------------------------------ fitting
    def _window(self, panel: pd.DataFrame, start, end) -> pd.DataFrame:
        P = panel.loc[start:end]
        if self.spec.window_years:
            first = P.index.max() - pd.DateOffset(years=self.spec.window_years) + pd.DateOffset(months=1)
            P = P.loc[first:]
        return P

    def target_columns(self) -> list[str]:
        return [f"{self.spec.target}_{a}" for a in self.assets]

    def fit(self, panel: pd.DataFrame, start=config.MODEL_START, end=None,
            honor_asset_start: bool = True) -> "AssetFactorModel":
        """Fit on the months of [start, end] where every feature and every model asset's return exist.

        An asset's asset_start is honoured when the months from it to the end of the window number at least
        MIN_ASSET_WINDOW; otherwise, or with honor_asset_start=False (per-era betas), the asset uses the
        common start and asset_start_used records None for it.
        """
        if self.spec.kind == "mlp" and (asset_extras(self.spec) or self.spec.asset_start):
            raise ValueError(f"{self.spec.name}: per-asset factors and fit windows need a linear transmission "
                             "(the neural network is report-only)")
        X_all = design_matrix(panel, self.spec)
        P = self._window(panel, start, end)
        X = X_all.loc[P.index]
        Y = P[self.target_columns()]
        mask = X.notna().all(axis=1) & Y.notna().all(axis=1)
        X, Y = X[mask], Y[mask]
        if len(X) < 3 * (X.shape[1] + 2):
            raise ValueError(f"not enough months to fit {self.spec.name}: {len(X)}")
        self.train_start, self.train_end = X.index.min(), X.index.max()
        x = X.to_numpy(float)
        y = Y.to_numpy(float)
        self.x_mean = x.mean(axis=0)
        self.x_sd = x.std(axis=0)
        self.x_sd[self.x_sd < 1e-12] = 1.0
        z = (x - self.x_mean) / self.x_sd
        own = self._own_rows(X.index, honor_asset_start)

        if self.spec.kind == "mlp":
            self._fit_mlp(z, y)
        else:
            self._fit_linear(x, z, y, own)

        pred = self._predict_z(z)
        res = y - pred
        self.asset_train_start = {}
        for j, a in enumerate(self.assets):
            rows = own.get(a)
            yj, rj = (y[:, j], res[:, j]) if rows is None else (y[rows, j], res[rows, j])
            sst = ((yj - yj.mean()) ** 2).sum()
            self.stats.r2[a] = float(1.0 - (rj ** 2).sum() / sst) if sst > 0 else 0.0
            self.stats.n_obs[a] = int(len(yj))
            self.stats.resid_sd[a] = float(rj.std())
            self.asset_train_start[a] = (X.index[0] if rows is None else X.index[rows][0]).strftime("%Y-%m")
        return self

    def _own_rows(self, index: pd.DatetimeIndex, honor: bool) -> dict[str, np.ndarray]:
        """Row masks (over the fit months) of the assets that start later on their own; records
        asset_start_used for every asset with an asset_start."""
        self.asset_start_used = {}
        out = {}
        for a, month in self.spec.asset_start:
            if a not in self.assets:
                continue
            rows = np.asarray(index >= pd.Timestamp(month))
            used = honor and int(rows.sum()) >= MIN_ASSET_WINDOW
            self.asset_start_used[a] = month if used else None
            if used and not rows.all():
                out[a] = rows
        return out

    def _fit_linear(self, x: np.ndarray, z: np.ndarray, y: np.ndarray, own: dict[str, np.ndarray]) -> None:
        extras = asset_extras(self.spec)
        start = dict(self.spec.asset_start)
        groups: dict[tuple, list[int]] = {}
        for j, a in enumerate(self.assets):
            groups.setdefault((extras.get(a, ()), start[a] if a in own else None), []).append(j)
        if list(groups) == [((), None)] and not _extra_ids(self.spec):
            # one equation for every asset on the joint design: the model as it always was (not when the spec
            # has extras that none of the modelled assets owns: those stay out of every equation)
            coef, intercept, tstat = _linear(z, y, self.x_mean, self.x_sd, self.spec.ridge)
            self.coef = pd.DataFrame(coef.T, index=self.assets, columns=self.features)
            self.intercept = pd.Series(intercept, index=self.assets)
            self.tstat = pd.DataFrame(tstat.T, index=self.assets, columns=self.features)
            return
        pos = {f: i for i, f in enumerate(self.features)}
        extra = set(_extra_ids(self.spec))
        common = [i for i, f in enumerate(self.features) if f not in extra]
        coef = np.zeros((len(self.assets), len(self.features)))
        tstat = np.full_like(coef, np.nan)
        intercept = np.zeros(len(self.assets))
        for (ids, first), js in groups.items():
            cols = common + [pos[f] for f in ids]
            rows = own[self.assets[js[0]]] if first else np.ones(len(x), bool)
            xg = x[np.ix_(rows, cols)]
            if len(xg) < 3 * (len(cols) + 2):
                raise ValueError(f"not enough months to fit {self.spec.name} for {self.assets[js[0]]}: {len(xg)}")
            mean, sd = xg.mean(axis=0), xg.std(axis=0)
            sd[sd < 1e-12] = 1.0
            c, b, t = _linear((xg - mean) / sd, y[np.ix_(rows, js)], mean, sd, self.spec.ridge)
            coef[np.ix_(js, cols)] = c.T
            tstat[np.ix_(js, cols)] = t.T
            intercept[js] = b
        self.coef = pd.DataFrame(coef, index=self.assets, columns=self.features)
        self.intercept = pd.Series(intercept, index=self.assets)
        self.tstat = pd.DataFrame(tstat, index=self.assets, columns=self.features)

    def _fit_mlp(self, z: np.ndarray, y: np.ndarray) -> None:
        from sklearn.neural_network import MLPRegressor

        self.y_mean = y.mean(axis=0)
        self.y_sd = y.std(axis=0)
        self.y_sd[self.y_sd < 1e-12] = 1.0
        yz = (y - self.y_mean) / self.y_sd
        self.mlp = []
        for seed in range(3):  # a tiny ensemble to tame seed variance
            m = MLPRegressor(hidden_layer_sizes=(self.spec.hidden,), activation="tanh",
                             alpha=1e-2, learning_rate_init=3e-3, max_iter=3000,
                             early_stopping=True, validation_fraction=0.15,
                             n_iter_no_change=40, random_state=seed)
            m.fit(z, yz)
            self.mlp.append(m)

    # ------------------------------------------------------------ prediction
    def _predict_z(self, z: np.ndarray) -> np.ndarray:
        if self.spec.kind == "mlp":
            out = np.mean([m.predict(z) for m in self.mlp], axis=0)
            out = np.atleast_2d(out)
            if out.shape[0] != z.shape[0]:
                out = out.T
            return out * self.y_sd + self.y_mean
        x = z * self.x_sd + self.x_mean
        return x @ self.coef.to_numpy().T + self.intercept.to_numpy()

    def predict_array(self, x: np.ndarray) -> np.ndarray:
        """Systematic return for raw feature rows (n, k) -> (n, assets).

        Linear: x @ coef.T + intercept (coef is in original units for every equation; the round trip
        through the joint standardization only keeps the numbers bit for bit what they always were)."""
        z = (np.asarray(x, float) - self.x_mean) / self.x_sd
        return self._predict_z(z)

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Systematic (factor-explained) return for every month with features."""
        X = design_matrix(panel, self.spec)
        ok = X.notna().all(axis=1)
        out = pd.DataFrame(np.nan, index=panel.index, columns=self.assets)
        if ok.any():
            out.loc[ok] = self.predict_array(X[ok].to_numpy(float))
        return out

    def residuals(self, panel: pd.DataFrame) -> pd.DataFrame:
        actual = panel[self.target_columns()].copy()
        actual.columns = self.assets
        return actual - self.predict(panel)

    def contributions(self, panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Per-asset monthly contribution of each base factor (linear models only).

        Lag and regime terms are folded into their base factor so the
        attribution reads "how much did oil explain", not "oil_l2". Every
        asset gets a column per factor of the union; one outside its equation
        contributes 0.
        """
        if self.coef is None:
            raise ValueError("contributions are only defined for linear models")
        X = design_matrix(panel, self.spec)
        base = [c[2:] if c.startswith("f_") else c for c in base_columns(self.spec)]
        out = {}
        for a in self.assets:
            contrib = {}
            for b in base:
                cols = [c for c in self.features if c == b or c.startswith(b + "_l") or c == b + "_hi"]
                contrib[b] = (X[cols] * self.coef.loc[a, cols]).sum(axis=1, min_count=1)
            out[a] = pd.DataFrame(contrib)
        return out

    def summary(self) -> dict:
        d = {
            "spec": self.spec.to_dict(),
            "describe": self.spec.describe(),
            "train_start": str(self.train_start.date()) if self.train_start is not None else None,
            "train_end": str(self.train_end.date()) if self.train_end is not None else None,
            "r2": self.stats.r2,
            "resid_sd": self.stats.resid_sd,
            "asset_factors": {a: list(ids) for a, ids in asset_extras(self.spec).items()},
            "asset_start": dict(self.spec.asset_start),
            "asset_train_start": dict(self.asset_train_start),
            "asset_start_used": dict(self.asset_start_used),
        }
        if self.coef is not None:
            d["coef"] = {a: {k: float(v) for k, v in self.coef.loc[a].items()} for a in self.assets}
            d["tstat"] = {a: {k: float(v) for k, v in self.tstat.loc[a].items()} for a in self.assets}
            d["intercept"] = {a: float(v) for a, v in self.intercept.items()}
        return d
