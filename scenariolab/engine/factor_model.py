"""Factor -> asset transmission models.

Each asset's monthly price return is explained by the same month's macro factor
changes (contemporaneous transmission), optionally with lags, an
inflation-regime interaction, extra candidate factors / structural shocks, or a
small neural network instead of a linear map. The diagnostics compare these
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
    name: str = "base"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["factors"] = list(self.factors)
        d["shock_cols"] = list(self.shock_cols)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
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
        return "、".join(parts)


def base_columns(spec: ModelSpec) -> list[str]:
    return [f"f_{f}" for f in spec.factors] + list(spec.shock_cols)


def design_matrix(panel: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Feature matrix for every month of `panel` (NaN where lags are missing)."""
    base = panel[base_columns(spec)].copy()
    base.columns = [c[2:] if c.startswith("f_") else c for c in base.columns]
    parts = [base]
    for lag in range(1, spec.lags + 1):
        parts.append(base.shift(lag).add_suffix(f"_l{lag}"))
    if spec.regime == "inflation":
        dummy = (panel["s_cpi_yoy"].shift(1) > HIGH_INFLATION).astype(float)
        inter = base.mul(dummy, axis=0).add_suffix("_hi")
        parts.append(inter)
        parts.append(dummy.rename("hi_infl").to_frame())
    return pd.concat(parts, axis=1)


def feature_names(spec: ModelSpec) -> list[str]:
    names = [c[2:] if c.startswith("f_") else c for c in base_columns(spec)]
    out = list(names)
    for lag in range(1, spec.lags + 1):
        out += [f"{n}_l{lag}" for n in names]
    if spec.regime == "inflation":
        out += [f"{n}_hi" for n in names] + ["hi_infl"]
    return out


def features_from_paths(spec: ModelSpec, base_paths: np.ndarray, base_init: np.ndarray,
                        hi_infl: np.ndarray | None) -> np.ndarray:
    """Build the feature tensor for simulated paths.

    base_paths: (n_paths, T, n_base) simulated base factor changes in spec order.
    base_init:  (lags, n_base) the last actual months before the simulation
                start, oldest first, used to fill lags at the start of a path.
    hi_infl:    (n_paths, T) high-inflation dummy (needed if spec.regime).
    """
    n, T, k = base_paths.shape
    parts = [base_paths]
    if spec.lags:
        init = np.broadcast_to(base_init[None, :, :], (n, spec.lags, k))
        ext = np.concatenate([init, base_paths], axis=1)          # (n, lags+T, k)
        for lag in range(1, spec.lags + 1):
            parts.append(ext[:, spec.lags - lag: spec.lags - lag + T, :])
    if spec.regime == "inflation":
        d = hi_infl[:, :, None]
        parts.append(base_paths * d)
        parts.append(d)
    return np.concatenate(parts, axis=2)


@dataclass
class FitStats:
    r2: dict = field(default_factory=dict)
    n_obs: dict = field(default_factory=dict)
    resid_sd: dict = field(default_factory=dict)


class AssetFactorModel:
    """One spec, fitted per asset on a training window."""

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

    # ------------------------------------------------------------ fitting
    def _window(self, panel: pd.DataFrame, start, end) -> pd.DataFrame:
        P = panel.loc[start:end]
        if self.spec.window_years:
            first = P.index.max() - pd.DateOffset(years=self.spec.window_years) + pd.DateOffset(months=1)
            P = P.loc[first:]
        return P

    def target_columns(self) -> list[str]:
        return [f"{self.spec.target}_{a}" for a in self.assets]

    def fit(self, panel: pd.DataFrame, start=config.MODEL_START, end=None) -> "AssetFactorModel":
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

        if self.spec.kind == "mlp":
            self._fit_mlp(z, y)
        else:
            self._fit_linear(z, y, X.index)

        pred = self._predict_z(z)
        res = y - pred
        for j, a in enumerate(self.assets):
            sst = ((y[:, j] - y[:, j].mean()) ** 2).sum()
            self.stats.r2[a] = float(1.0 - (res[:, j] ** 2).sum() / sst) if sst > 0 else 0.0
            self.stats.n_obs[a] = int(len(y))
            self.stats.resid_sd[a] = float(res[:, j].std())
        return self

    def _fit_linear(self, z: np.ndarray, y: np.ndarray, index) -> None:
        n, k = z.shape
        ym = y.mean(axis=0)
        A = z.T @ z + self.spec.ridge * n * np.eye(k)
        beta_z = np.linalg.solve(A, z.T @ (y - ym))           # (k, assets)
        coef = beta_z / self.x_sd[:, None]
        intercept = ym - self.x_mean @ coef
        self.coef = pd.DataFrame(coef.T, index=self.assets, columns=self.features)
        self.intercept = pd.Series(intercept, index=self.assets)
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
        self.tstat = pd.DataFrame((beta_z / se).T, index=self.assets, columns=self.features)

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
        """Systematic return for raw feature rows (n, k) -> (n, assets)."""
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
        attribution reads "how much did oil explain", not "oil_l2".
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
        }
        if self.coef is not None:
            d["coef"] = {a: {k: float(v) for k, v in self.coef.loc[a].items()} for a in self.assets}
            d["tstat"] = {a: {k: float(v) for k, v in self.tstat.loc[a].items()} for a in self.assets}
            d["intercept"] = {a: float(v) for a, v in self.intercept.items()}
        return d
