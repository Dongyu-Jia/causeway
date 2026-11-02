"""Policy reaction functions, automatic stabilizers and the policy shocks u_* (SIM_DESIGN §5.1, §5.3; ports the
research qmodel's PolicySplit, schema-revise-2 3.0 v01).

Every policy field of the card is mostly a reaction to the economy (reaction-function R² 0.40-0.54, SIM_DESIGN E18).
The systematic part of each field is an OLS on what the recursive ordering allows -- same-quarter E0 macro and the
start-of-quarter states (config_q.REACTION; the regressors come from qmodel.panel.reaction_frame) -- fitted on
REACTION_START .. the train end on current-vintage data (ALFRED re-estimation is open, SIM_DESIGN §15.4). The shock
is the rest:

    u_ff    d_ff - reaction function; fitted on non-ZLB quarters only; u_ff = 0 in zero-bound quarters
    u_bs    d_bs - balance-sheet rule; fitted from 2008Q3; u_bs = 0 before (no balance-sheet policy)
    u_def   d_deficit - automatic stabilizer (same- and previous-quarter unemployment change, IP, equity returns)
    u_tr    d_transfers - automatic stabilizer
    u_ptax  d_perstax - includes the last four quarters' equity returns (capital-gains taxes)

What a shock is used for: the measured history of u_* (the agent tree's root state, T-A, reports), the unit of a
QTable `u` entry (a size bin is a multiple of the training sd of that shock, shock_sd), and the reference
reaction function the factor layer's structural identification is compared with. The factor layer itself
(qmodel.factors) identifies its policy shocks recursively inside the joint model ([E0] -> [E0p] -> [E1c, E1, E2]
within the quarter), so a stated u_* never moves same-quarter E0 macro.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config_q as cq
from . import panel as qpanel

FIELDS = list(cq.REACTION)                                  # d_ff, d_deficit, d_transfers, d_perstax, d_bs
SHOCK_OF = {f: spec["shock"] for f, spec in cq.REACTION.items()}
FIELD_OF = {u: f for f, u in SHOCK_OF.items()}
SHOCKS = [SHOCK_OF[f] for f in FIELDS]
NAMES = {"u_ff": "政策利率冲击", "u_bs": "资产负债表冲击", "u_def": "赤字冲击", "u_tr": "转移支付冲击",
         "u_ptax": "个人税冲击"}
MIN_OBS = 16                                                # fewer complete quarters: the equation is not identified


@dataclass
class ReactionFit:
    """One systematic policy equation fitted to a train end."""
    field: str
    shock: str
    sample_rule: str
    regressors: list[str]
    train_end: str
    coef: pd.Series | None = None          # const + regressors (None: not identified at this train end)
    r2: float = float("nan")
    n: int = 0
    resid_sd: float = float("nan")          # training sd of the shock (the QTable size unit)
    first: str | None = None
    last: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def identified(self) -> bool:
        return self.coef is not None

    def to_dict(self) -> dict:
        return {"field": self.field, "shock": self.shock, "name": NAMES[self.shock], "sample": self.sample_rule,
                "regressors": list(self.regressors), "train_end": self.train_end, "identified": self.identified,
                "coef": None if self.coef is None else {k: float(v) for k, v in self.coef.items()},
                "r2": None if not np.isfinite(self.r2) else float(self.r2), "n": int(self.n),
                "resid_sd": None if not np.isfinite(self.resid_sd) else float(self.resid_sd),
                "first": self.first, "last": self.last,
                "check": {k: cq.REACTION[self.field][k].get(self.train_end) for k in ("check_r2", "check_n")}}


def _window(Q: pd.DataFrame, train_end: str) -> np.ndarray:
    return ((Q.index >= pd.Period(cq.REACTION_START, "Q")) & (Q.index <= pd.Period(train_end, "Q")))


def zlb_quarters(Q: pd.DataFrame) -> pd.Series:
    """1 where the policy rate sat below config_q.ZLB_FF at both the start and the end of the quarter (the
    reaction-function sample rule; u_ff = 0 there)."""
    ff_end = Q["s_ff"].shift(-1).fillna(Q["s_ff"] + Q["d_ff"])
    return ((Q["s_ff"] < cq.ZLB_FF) & (ff_end < cq.ZLB_FF)).astype(float)


def fit_one(Q: pd.DataFrame, field_id: str, train_end: str) -> ReactionFit:
    spec = cq.REACTION[field_id]
    y, X, sample = qpanel.reaction_frame(Q, field_id)
    fit = ReactionFit(field=field_id, shock=spec["shock"], sample_rule=spec["sample"], regressors=list(X.columns),
                      train_end=str(pd.Period(train_end, "Q")))
    keep = _window(Q, train_end) & sample.to_numpy(bool)
    df = pd.concat([y.rename("_y"), X], axis=1)[keep].dropna()
    if len(df) < MIN_OBS:
        fit.n = int(len(df))
        return fit
    Z = np.column_stack([np.ones(len(df)), df.drop(columns="_y").to_numpy(float)])
    yy = df["_y"].to_numpy(float)
    b, *_ = np.linalg.lstsq(Z, yy, rcond=None)
    e = yy - Z @ b
    fit.coef = pd.Series(b, index=["const"] + list(X.columns))
    fit.r2 = float(1.0 - (e @ e) / ((yy - yy.mean()) @ (yy - yy.mean())))
    fit.n = int(len(df))
    fit.resid_sd = float(np.sqrt((e @ e) / max(len(e) - Z.shape[1], 1)))
    fit.first, fit.last = str(df.index.min()), str(df.index.max())
    return fit


def fit(Q: pd.DataFrame, train_end: str) -> dict[str, ReactionFit]:
    """The five systematic equations fitted on REACTION_START .. train_end (quarters <= train_end only)."""
    return {f: fit_one(Q, f, train_end) for f in FIELDS}


def systematic(Q: pd.DataFrame, fits: dict[str, ReactionFit]) -> pd.DataFrame:
    """The systematic part of each policy field in every quarter with its regressors (NaN otherwise). d_ff's is
    censored so the policy rate cannot fall below 0 (the reaction cannot cut through the zero bound)."""
    out = pd.DataFrame(index=Q.index)
    for f, ft in fits.items():
        _, X, _ = qpanel.reaction_frame(Q, f)
        if not ft.identified:
            out[f] = np.nan
            continue
        v = ft.coef["const"] + X[ft.regressors].to_numpy(float) @ ft.coef[ft.regressors].to_numpy(float)
        if f == "d_ff":
            v = np.maximum(v, -Q["s_ff"].to_numpy(float))
        out[f] = v
    return out


def shocks(Q: pd.DataFrame, fits: dict[str, ReactionFit]) -> pd.DataFrame:
    """u_* = field - systematic part; u_ff = 0 in zero-bound quarters, u_bs = 0 before BS_POLICY_START; an
    unidentified equation's shock is 0 where its field is observed. NaN where the field is missing."""
    sysp = systematic(Q, fits)
    zlb = zlb_quarters(Q).to_numpy(bool)
    out = pd.DataFrame(index=Q.index)
    for f, ft in fits.items():
        raw = Q[f]
        if not ft.identified:
            out[ft.shock] = np.where(raw.notna(), 0.0, np.nan)
            continue
        u = raw - sysp[f]
        if f == "d_ff":
            u = u.where(~zlb, 0.0).where(raw.notna())
        if f == "d_bs":
            u = u.where(Q.index >= pd.Period(cq.BS_POLICY_START, "Q"), 0.0).where(raw.notna())
        out[ft.shock] = u
    return out


def shock_sd(fits: dict[str, ReactionFit]) -> dict[str, float]:
    """Training sd of each shock (the residual sd of its equation): the unit of a QTable u entry's size bins
    (小 / 温和 / 明显 / 大幅 / 极端 = 0.25 / 0.5 / 1 / 2 / 3 sd)."""
    return {ft.shock: ft.resid_sd for ft in fits.values()}


def summary(fits: dict[str, ReactionFit]) -> dict:
    return {ft.shock: ft.to_dict() for ft in fits.values()}
