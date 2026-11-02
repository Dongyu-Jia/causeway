"""The missing-factor audit (缺失因子审计): what the research round tested, what changed the model, what is left.

    python -m scenariolab.analysis.factor_audit        # writes data/results/factor_audit.json (about 20 s)

The user's question was whether the forecasts miss because specific factors were never modelled. A research round
(2026-11-01) tested 476 same-month transmission candidates and 369 known-at forecast states in six families; an
adversarial verifier re-derived every adopted number. Its results are frozen in the package's research archive
(audit_archive/research_2026_11_01.json, reports in docs/research/2026-11-01-factor-audit/): every archived number
carries the tag 「研究轮存档（2026-11-01，面板 3869fdde3437）」 and is never presented as recomputed.

The step runs in two modes on today's panel:

  replication  rebuilds the research round's base model (B0) from the panel's audit-only columns x_spread_mixed and
               x_labor_share_interp (and the old spread as the KEY_STATES spread state), re-runs the registered
               candidates (factor_tests: same-sample transmission, Stouffer z, refutation battery, circular placebo;
               three forecast states) and checks them against the archive: deltas to 1e-6, z to 1e-4, counts exactly.
  current      evaluates against today's transmission (model_selection.json, the walk-forward of diagnostics.json):
               each change's leave-one-out contribution, the residual anatomy, out-of-universe industries, coverage
               of the new columns and today's readings.

It needs diagnostics.json (the walk-forward) and reads nothing outside data/ and the package.
"""
from __future__ import annotations

import hashlib
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, jsonio
from ..engine.factor_model import MIN_ASSET_WINDOW, ModelSpec, asset_extras
from . import factor_tests as T

VERSION = 1
ARCHIVE_ID = "research_2026_11_01"
ARCHIVE_PATH = Path(__file__).with_name("audit_archive") / f"{ARCHIVE_ID}.json"
TOL_DELTA = 1e-6           # replication tolerance on archived R^2 changes
TOL_Z = 1e-4               # and on archived t / Stouffer z (the archive's inputs were stored with 6 digits)
B0_FACTORS = tuple(config.CORE_FACTOR_IDS)                    # the research round's base: the 10 core factors
DECS = ("T1", "T2", "T3")
OOU_INDUSTRIES = ["Durbl", "Chems", "BusEq", "Telcm", "Shops", "Other"]   # Fama-French industries the project does not use
OOU_NAMES = {"Durbl": "耐用消费品", "Chems": "化工", "BusEq": "商用设备", "Telcm": "电信", "Shops": "零售批发",
             "Other": "其他"}
FIRST_SIM_MONTH = "1973-02"
# the term-premium term's switch: {factor, state, lag, above} of config.FACTORS (today's readings report it)
TP_RULE = next(f["interaction"] for f in config.FACTORS if f["id"] == "tp10_x_core4")
ASSET_NAMES = {a["id"]: a["name"] for a in config.ASSETS}

# Candidates the replication re-runs: the panel column (or a function of the panel), the archive key, the
# assets the archive's battery covers and the base factors it replaces.
REGISTERED = [
    {"id": "spread_fixed", "archive": "risk:spread_fixed", "column": "f_spread", "drop": ("spread",),
     "name": "信用利差计时修正", "required": True},
    {"id": "gz", "archive": "risk:gz", "column": "f_gz", "drop": (), "name": "GZ 信用利差", "required": True},
    {"id": "tp10_x_core4", "archive": "rates:tp10_x_core4", "column": "f_tp10_x_core4", "drop": (),
     "name": "期限溢价 × 高核心通胀", "required": True},
    {"id": "metals_oil", "archive": "commod:metals_oil", "column": "f_metals_oil", "drop": (),
     "name": "基础金属相对油价（能源股）", "required": True},
    {"id": "tp10_x_infl4", "archive": "rates:tp10_x_infl4", "column": "f_tp10_x_infl4", "drop": (),
     "name": "期限溢价 × 上月 CPI > 4%", "required": False},
    {"id": "mortgage_spread", "archive": "risk:mortgage_spread", "column": None, "drop": (),
     "name": "房贷利差（黄金）", "required": False},
]
# Forecast states the replication re-runs (known-at alignment as in the research round)
PREDICTIVE = [
    {"id": "re_loans", "archive": "rates:re_loans__known_yoy -> cpi_yoy", "target": "cpi_yoy",
     "name": "银行房地产贷款同比 → CPI 同比变化", "lag": 1, "column": "s_re_loans_yoy", "log": False},
    {"id": "labor_share_rt", "archive": "labor:labor_share_rt -> cpi_yoy", "target": "cpi_yoy",
     "name": "劳动收入占比（按发布月，滞后 4 个月）→ CPI 同比变化", "lag": 0, "column": "s_labor_share", "log": True},
    {"id": "philly_capex", "archive": "tech:philly_capex.known_level -> ret_industrials", "target": "ret_industrials",
     "name": "费城联储资本开支计划 → 工业股未来 12 个月回报", "lag": 0, "column": "s_philly_capex", "log": False},
]


# ============================================================================ inputs

def load_archive(path: Path | None = None) -> dict:
    return jsonio.load(path or ARCHIVE_PATH)


def _sha(path: Path) -> str | None:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12] if Path(path).exists() else None


def research_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """The panel as the research round saw the base model: the spread with the mixed pre-1986 timing and the
    labor share interpolated between quarters (the audit-only x_ columns), also as the KEY_STATES spread state."""
    P = T.month_start(panel)
    P["s_spread"] = P["x_spread_mixed"]
    P["f_spread"] = P["x_spread_mixed"].diff()
    P["s_labor_share"] = P["x_labor_share_interp"]
    P["f_labor_share"] = np.log(P["x_labor_share_interp"]).diff()
    return P


def _series(P: pd.DataFrame, reg: dict) -> pd.Series:
    if reg["id"] == "mortgage_spread":
        return P["s_mort_spread"].diff()
    return P[reg["column"]]


def _names() -> dict:
    out = {f"f_{f['id']}": f["name"] for f in config.FACTORS}
    out.update({f"s_{s['id']}": s["name"] for s in config.STATES})
    out.update({"x_spread_mixed": "信用利差（旧计时，审计用）", "x_labor_share_interp": "劳动收入占比（季度插值，审计用）"})
    return out


# ============================================================================ replication mode

def _cmp(recomputed: float, archived, tol: float) -> float | None:
    if archived is None or recomputed is None or not np.isfinite(recomputed):
        return None
    return abs(float(recomputed) - float(archived))


def replicate(panel: pd.DataFrame, archive: dict, placebo_n: int = T.PLACEBO_N) -> dict:
    """Re-run the registered candidates on the research base and compare with the archive."""
    P = T.month_start(panel)
    P0 = research_panel(P)
    bench = T.Bench(P0, ModelSpec(factors=B0_FACTORS))
    out, checks = {}, {"max_delta_diff": 0.0, "max_t_diff": 0.0, "max_z_diff": 0.0, "count_mismatches": [],
                       "required_max_delta_diff": 0.0, "required_max_z_diff": 0.0, "required_max_t_diff": 0.0}
    for reg in REGISTERED:
        arc = archive["battery"].get(reg["archive"])
        if arc is None or (reg["column"] and reg["column"] not in P.columns) or \
                (reg["id"] == "mortgage_spread" and "s_mort_spread" not in P.columns):
            out[reg["id"]] = {"available": False}
            continue
        assets = list(arc["assets"])
        bat = T.refutation(bench, reg["id"], _series(P, reg), assets, reg["drop"], placebo_n)
        dd, dt, dz = 0.0, 0.0, 0.0
        for sec in ("base", "ex_crisis_score", "rolling", "lag1", "lead1"):
            for a in assets:
                for d, v in arc[sec].get(a, {}).items():
                    r = bat[sec][a].get(d)
                    if r is None:
                        checks["count_mismatches"].append(f"{reg['id']} {sec} {a} {d}: missing")
                        continue
                    dd = max(dd, _cmp(r["delta_ss"], v["delta_ss"], TOL_DELTA) or 0.0)
                    dt = max(dt, _cmp(r["dm_t"], v["dm_t_ss"], TOL_Z) or 0.0)
                    if r["n_train"] != v["n_train"]:
                        checks["count_mismatches"].append(f"{reg['id']} {sec} {a} {d}: n_train {r['n_train']} vs {v['n_train']}")
            for d, v in arc[sec].get("common", {}).items():
                r = bat[sec]["common"].get(d)
                if r is None:
                    continue
                dd = max(dd, _cmp(r["score_ss"], v["score_ss"], TOL_DELTA) or 0.0)
                dt = max(dt, _cmp(r["pooled_t"], v["pooled_t_ss"], TOL_Z) or 0.0)
        for a, (z, k) in bat["z"].items():
            za, ka = arc["z"][a]
            dz = max(dz, _cmp(z, za, TOL_Z) or 0.0)
            if k != ka:
                checks["count_mismatches"].append(f"{reg['id']} z {a}: {k} decades vs {ka}")
        rebuilt = archive.get("rederived", {}).get("retest_rebuilt", {}).get(reg["id"], {})
        dz_rebuilt = max([abs(bat["z"][a][0] - v["z"]) for a, v in rebuilt.items() if a in bat["z"]] or [float("nan")])
        out[reg["id"]] = {"available": True, "name": reg["name"], "archive_key": reg["archive"], "assets": assets,
                          "drop": list(reg["drop"]), "battery": bat,
                          "diff": {"max_delta": dd, "max_t": dt, "max_z": dz, "max_z_vs_rebuilt": dz_rebuilt},
                          "z_archive": {a: arc["z"][a][0] for a in arc["z"]},
                          "placebo_p_archive": {a: v.get("p_emp") for a, v in arc["placebo_circular"].items()}}
        checks["max_delta_diff"] = max(checks["max_delta_diff"], dd)
        checks["max_t_diff"] = max(checks["max_t_diff"], dt)
        checks["max_z_diff"] = max(checks["max_z_diff"], dz)
        if reg["required"]:
            checks["required_max_delta_diff"] = max(checks["required_max_delta_diff"], dd)
            checks["required_max_t_diff"] = max(checks["required_max_t_diff"], dt)
            checks["required_max_z_diff"] = max(checks["required_max_z_diff"], dz)

    # the B0 base itself (per asset and decade) against the design check values
    b0 = archive["design"]["d01_base_r2"]["0.0"]
    b0_diff = 0.0
    for j, key in enumerate(b0["columns"]):
        res = T.spec_score(P0, ModelSpec(factors=B0_FACTORS), *T.decade_window(key, P0), assets=b0["index"])
        b0_diff = max(b0_diff, max(abs(res["per_asset"][a] - b0["data"][i][j]) for i, a in enumerate(b0["index"])))
    checks["b0_max_diff"] = b0_diff

    # active months of the high-core-inflation dummy (counts are exact)
    dummy = (P["s_core_yoy"].shift(1) > 4.0).astype(float).where(P["s_core_yoy"].shift(1).notna())
    act = {}
    for k in DECS:
        d = T.DECADES[k]
        act[k] = {"train_active": int(dummy.loc[d["train"][0]:d["train"][1]].sum()),
                  "test_active": int(dummy.loc[d["test"][0]:d["test"][1]].sum())}
    arc_act = archive["rederived"]["regime_active_months_core4"]
    for k in DECS:
        if act[k] != arc_act[k]:
            checks["count_mismatches"].append(f"core4 active months {k}: {act[k]} vs {arc_act[k]}")

    # forecast states
    pred_rows = []
    cands = {}
    for p in PREDICTIVE:
        if p["column"] not in P.columns:
            continue
        x = np.log(P[p["column"]]) if p["log"] else P[p["column"]]
        cands[p["id"]] = x.shift(p["lag"]) if p["lag"] else x
    tab = T.predictive_test(P0, cands, sorted({p["target"] for p in PREDICTIVE})) if cands else pd.DataFrame()
    pdiff = 0.0
    for p in PREDICTIVE:
        if p["id"] not in cands:
            continue
        arc = archive["forecast_states"]["replicated_archive"][p["archive"]]["base"]
        m = tab[(tab["candidate"] == p["id"]) & (tab["target"] == p["target"]) & (tab["origins"] == "monthly")]
        m = m.set_index("period")
        row = {"id": p["id"], "name": p["name"], "target": p["target"], "archive_key": p["archive"],
               "r2_incr": [float(m.loc[per, "r2_incr"]) for per in ("all", "h1", "h2")],
               "cw_p": float(m.loc["all", "cw_p"]), "dm_p": float(m.loc["all", "dm_p"]),
               "n": [int(m.loc[per, "n"]) for per in ("all", "h1", "h2")],
               "r2_incr_archive": [arc[per]["r2_incr"] for per in ("all", "h1", "h2")],
               "cw_p_archive": arc["all"]["cw_p"], "n_archive": [arc[per]["n"] for per in ("all", "h1", "h2")]}
        ye = tab[(tab["candidate"] == p["id"]) & (tab["target"] == p["target"]) & (tab["origins"] == "year_end")]
        row["year_end_r2_incr"] = [float(v) for v in ye.set_index("period").reindex(["all", "h1", "h2"])["r2_incr"]]
        row["diff"] = max(abs(a - b) for a, b in zip(row["r2_incr"], row["r2_incr_archive"]))
        if row["n"] != row["n_archive"]:
            checks["count_mismatches"].append(f"forecast state {p['id']}: n {row['n']} vs {row['n_archive']}")
        pdiff = max(pdiff, row["diff"])
        pred_rows.append(row)
    checks["predictive_max_diff"] = pdiff
    checks["required"] = [r["id"] for r in REGISTERED if r["required"]]
    checks["tolerance"] = {"delta": TOL_DELTA, "z": TOL_Z, "counts": "exact"}
    # the same differences as text: results files round every float to 6 decimals, which would show the
    # replication's worst R^2 difference (about 5e-7) as an exact 0.0
    checks["sci"] = {k: f"{checks[k]:.2e}" for k in ("max_delta_diff", "required_max_delta_diff", "max_t_diff",
                                                     "required_max_t_diff", "max_z_diff", "required_max_z_diff",
                                                     "b0_max_diff", "predictive_max_diff")}
    checks["ok"] = bool(checks["required_max_delta_diff"] <= TOL_DELTA and checks["required_max_z_diff"] <= TOL_Z
                        and checks["required_max_t_diff"] <= TOL_Z and checks["predictive_max_diff"] <= TOL_DELTA
                        and not checks["count_mismatches"] and checks["b0_max_diff"] <= 1e-4)
    return {"candidates": out, "predictive": pred_rows, "core4_active_months": act, "checks": checks}


# ============================================================================ current mode

def _common_spec(spec: ModelSpec) -> ModelSpec:
    """The selected spec without its per-asset equations (the common transmission of stage A)."""
    return ModelSpec.from_dict({**spec.to_dict(), "asset_factors": {}, "asset_start": {}})


def _with(spec: ModelSpec, **change) -> ModelSpec:
    return ModelSpec.from_dict({**spec.to_dict(), **change})


def _decade_scores(P: pd.DataFrame, specs: dict, assets=None) -> dict:
    """{name: {decade: {score, per_asset, n_months}}} with every spec scored on the months all of them predict."""
    out = {k: {} for k in specs}
    for d in DECS:
        res = T.compare(P, specs, *T.decade_window(d, P), assets=assets)
        for k in specs:
            out[k][d] = {"score": res[k]["score"], "per_asset": res[k]["per_asset"], "n_months": res[k]["n_months"]}
    return out


def _loo(P: pd.DataFrame, spec: ModelSpec) -> dict:
    """Each change's contribution in `spec`: its score (or its asset's R^2) minus the score without it, per decade.
    None for a change that is not in the spec."""
    extras = asset_extras(spec)
    starts = dict(spec.asset_start)
    alts = {}
    for f in ("gz", "tp10_x_core4"):
        if f in spec.factors:
            alts[f] = _with(spec, factors=[x for x in spec.factors if x != f])
    if "labor_share" not in spec.factors:          # dropping it: the score without it minus the score with it
        alts["labor_share"] = _with(spec, factors=list(spec.factors) + ["labor_share"])
    if "metals_oil" in extras.get("energy", ()):
        alts["metals_oil"] = _with(spec, asset_factors={a: list(v) for a, v in extras.items() if a != "energy"})
    if "gold" in starts:
        alts["gold_window"] = _with(spec, asset_start={a: v for a, v in starts.items() if a != "gold"})
    out = {}
    for k, alt in alts.items():
        res = _decade_scores(P, {"with": spec, "without": alt})
        asset = {"metals_oil": "energy", "gold_window": "gold"}.get(k)
        if asset:
            out[k] = [res["with"][d]["per_asset"][asset] - res["without"][d]["per_asset"][asset] for d in DECS]
        else:
            out[k] = [res["with"][d]["score"] - res["without"][d]["score"] for d in DECS]
    return out


def _spread_fix_in(P: pd.DataFrame, spec: ModelSpec) -> list | None:
    """The spread timing fix inside `spec`: its score minus the score with the old mixed-timing spread."""
    if "spread" not in spec.factors:
        return None
    P_old = P.copy()
    P_old["f_spread"] = P["x_spread_mixed"].diff()
    out = []
    for d in DECS:
        new = T.spec_score(P, spec, *T.decade_window(d, P))["score"]
        old = T.spec_score(P_old, spec, *T.decade_window(d, P))["score"]
        out.append(new - old)
    return out


def residual_anatomy(P: pd.DataFrame, spec: ModelSpec, base_resid: pd.DataFrame | None = None) -> dict:
    """Expanding-window OOS residuals (refit every January, 1985-01..): PC1 share, its link to the market's own
    P/D change, the P/D R^2 of the us_equity residual, SSE kept against a base and the top-10-month share."""
    R = T.oos_residuals(P, spec)
    pca = T.residual_pca(R[T.SCORE_ASSETS])
    pdv = P["f_valuation"]
    t3 = R.loc["2015-01":, T.SCORE_ASSETS]
    out = {"months": pca["months"], "first": T.ym(R.index.min()), "last": T.ym(R.index.max()),
           "pc1_share": pca["share"][0], "pc_shares": pca["share"],
           "pc1_corr_us_equity_resid": pca["corr_with_us_equity_resid"][0],
           "pc1_corr_pd_change": float(pca["scores"]["PC1"].corr(pdv.reindex(pca["scores"].index))),
           "pd_r2_us_equity": T.r2_on(R["us_equity"], pdv),
           "pd_r2_us_equity_2015_2026": T.r2_on(R.loc["2015-01":, "us_equity"], pdv),
           "top10_share_2015_2026": {a: float((t3[a] ** 2).sort_values(ascending=False).iloc[:10].sum() / (t3[a] ** 2).sum())
                                     for a in T.SCORE_ASSETS},
           "loadings_pc1": pca["loadings"]["PC1"].to_dict()}
    if base_resid is not None:
        out["sse_kept_vs_base"] = {a: float((R[a] ** 2).sum() / (base_resid[a] ** 2).sum()) for a in T.SCORE_ASSETS}
    return out, R


def out_of_universe(P: pd.DataFrame, P0: pd.DataFrame, spec: ModelSpec) -> dict | None:
    """The six Fama-French industries the project does not use (total returns): the selected common factors
    against the research base, same windows and fit months as the walk-forward."""
    path = config.RAW_DIR / "french" / "12_Industry_Portfolios_CSV.zip"
    if not path.exists():
        return None
    from ..data.sources import _parse_french_tables
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        tabs = _parse_french_tables(zf.read(member).decode("latin-1"))
    ind = next(df for t, df in tabs.items() if t.startswith("Average Value Weighted Returns -- Monthly")) / 100.0
    base_f = [f"f_{f}" for f in B0_FACTORS]
    new_f = [f"f_{f}" for f in spec.factors] + list(spec.shock_cols)
    rows = {}
    for c in OOU_INDUSTRIES:
        if c not in ind.columns:
            continue
        b, s = T.external_r2(P0, base_f, ind[c]), T.external_r2(P, new_f, ind[c])
        rows[c] = {"name": OOU_NAMES[c], "base": [b[d] for d in DECS], "final_common": [s[d] for d in DECS]}
    if not rows:
        return None
    delta = [float(np.mean([r["final_common"][i] - r["base"][i] for r in rows.values()])) for i in range(3)]
    better = [int(sum(r["final_common"][i] > r["base"][i] for r in rows.values())) for i in range(3)]
    return {"industries": rows, "delta_vs_base": delta, "n_better": better, "n": len(rows),
            "spec": spec.describe(), "returns": "total"}


def coverage(P: pd.DataFrame, meta: dict) -> list:
    """The new model inputs, view variables and audit columns: timing, coverage since the MonthDB's first month."""
    info = meta.get("column_info", {}) or {}
    names = _names()
    new_factor_cols = {f"f_{f['id']}" for f in config.FACTORS if f["id"] in
                       {"gz", "tp10", "tp10_x_core4", "metals_oil", "tp10_x_infl4", "ebp"}}
    view_cols = {f"s_{s['id']}" for s in config.STATES if s.get("concept")}
    cols = [c for c in info if c in new_factor_cols or c in view_cols or c.startswith("x_")
            or c in ("s_spread", "f_spread", "s_labor_share", "f_labor_share")]
    out = []
    end = P.index.max()
    for c in cols:
        ci = info[c]
        s = P[c] if c in P.columns else pd.Series(dtype=float)
        win = s.loc[FIRST_SIM_MONTH:] if len(s) else s
        out.append({"col": c, "name": names.get(c, c), "role": ci.get("role"), "first": ci.get("first"),
                    "last": ci.get("last"), "share_since_1973_02": float(win.notna().mean()) if len(win) else 0.0,
                    "full_history": bool(len(win) and win.notna().all() and T.ym(s.dropna().index.max()) == T.ym(end)),
                    "alignment": ci.get("alignment"), "lag_months": ci.get("lag_months"), "revised": ci.get("revised"),
                    "copyright": ci.get("copyright"), "late": ci.get("late"), "sources": ci.get("sources")})
    return out


def today(P: pd.DataFrame, db_index: pd.DatetimeIndex | None) -> dict:
    """Today's readings of the new inputs (the overview line and the audit's caveats)."""
    last = P.index.max()
    since = P.loc[FIRST_SIM_MONTH:]                   # the month database's months (1973-02 on), as the design's d07

    def level(col):
        s = P[col].dropna() if col in P.columns else pd.Series(dtype=float)
        if s.empty:
            return None
        v = float(s.iloc[-1])
        hist = since[col].dropna()
        return {"value": v, "month": T.ym(s.index[-1]), "pct_since_1973": float((hist < v).mean()) if len(hist) else None,
                "max": float(hist.max()) if len(hist) else None, "max_month": T.ym(hist.idxmax()) if len(hist) else None}
    # the term-premium term's switch (config.FACTORS "interaction"): month t is on when the state `lag` months
    # earlier is above the threshold, so today's switch is decided by an earlier reading (core_yoy_prev)
    rule = TP_RULE
    state = P[f"s_{rule['state']}"]
    core_prev = state.shift(int(rule["lag"]))
    dummy = (core_prev > float(rule["above"])).where(core_prev.notna())
    months = db_index if db_index is not None else P.loc[FIRST_SIM_MONTH:].index
    d_db = dummy.reindex(months)
    on = dummy[dummy == True].index  # noqa: E712 - a float/NaN series
    decided = dummy.dropna().index.max() if dummy.notna().any() else None
    prev_month = decided - pd.DateOffset(months=int(rule["lag"])) if decided is not None else None   # monthly panel
    return {"month": T.ym(last), "gz": level("s_gz"), "ebp": level("s_ebp"), "tp10": level("s_tp10"),
            "spread": level("s_spread"), "core_yoy": float(P["s_core_yoy"].dropna().iloc[-1]),
            "core_yoy_month": T.ym(P["s_core_yoy"].dropna().index[-1]),
            "tp_term_rule": {"state": rule["state"], "lag": int(rule["lag"]), "above": float(rule["above"])},
            "tp_term_month": T.ym(decided) if decided is not None else None,
            "core_yoy_prev": float(core_prev.loc[decided]) if decided is not None else None,
            "core_yoy_prev_month": T.ym(prev_month) if prev_month is not None else None,
            "tp_term_on": bool(dummy.dropna().iloc[-1]) if dummy.notna().any() else None,
            "tp_term_active_months_monthdb": int(d_db.sum()), "monthdb_months": int(len(months)),
            "tp_term_active_share": float(d_db.mean()), "tp_term_last_on": T.ym(on.max()) if len(on) else None}


# ============================================================================ the walk-forward rows

def _walkforward(diag: dict, cid: str) -> dict:
    """What diagnostics.json's walk-forward did with one change (common: stage A; dedicated: stage B)."""
    its = []
    for it in diag.get("iterations", []) or []:
        sel = it.get("selected", {}) or {}
        path = sel.get("path") or []
        hit = next((p for p in path if p.get("change") == cid), None)
        r1 = next((t for t in it.get("trials", []) or [] if t.get("candidate") == cid and t.get("round") == 1), None)
        abl = next((t for t in it.get("test_ablation", []) or [] if t.get("candidate") == cid), None)
        tr = it.get("test_result", {}) or {}
        its.append({"iter": it.get("iter"), "selected": cid in (sel.get("changes") or []),
                    "round": hit.get("round") if hit else None, "val_score": hit.get("val_score") if hit else None,
                    "round1_delta": r1.get("delta") if r1 else None, "eligible": r1.get("eligible") if r1 else None,
                    "ablation": {k: abl.get(k) for k in ("val_delta", "test_delta", "test_months")} if abl else None,
                    "test_improved": tr.get("improved"), "test_delta": tr.get("delta"),
                    "test_delta_vs_start": tr.get("delta_vs_start")})
    sb = diag.get("stage_b") or {}
    rows = []
    for it in sb.get("iterations", []) or []:
        for r in it.get("rows", []) or []:
            if r.get("id") == cid:
                rows.append({"iter": it.get("iter"), **{k: r.get(k) for k in ("asset", "option", "val_gain", "test_gain",
                                                                                 "accepted")}})
    final = diag.get("final", {}) or {}
    return {"iterations": its, "stage_b": rows or None, "in_final": cid in (final.get("changes") or []),
            "accepted_stage_b": cid in (sb.get("accepted") or [])}


# ============================================================================ assembly

def _f(v, nd=3, sign=True):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _three(sec: dict, key: str) -> list | None:
    if key == "common":
        return [sec["common"][d]["score_ss"] for d in DECS] if all(d in sec.get("common", {}) for d in DECS) else None
    if key not in sec:
        return None
    return [sec[key][d]["delta_ss"] for d in DECS]


def _refutation_row(bat: dict, key: str) -> dict:
    roll = bat["rolling"]["common" if key == "common" else key]
    vals = [v["score_ss" if key == "common" else "delta_ss"] for v in roll.values()]
    return {"decades": _three(bat["base"], key), "ex_crisis": _three(bat["ex_crisis_score"], key),
            "rolling": {"positive": int(sum(v > 0 for v in vals)), "n": len(vals)},
            "lag": _three(bat["lag1"], key), "lead": _three(bat["lead1"], key),
            "placebo_p": bat["placebo_circular"][key]["p"], "z": bat["z"][key][0]}


def run(ctx) -> dict:
    archive = load_archive()
    diag_path = config.RESULTS_DIR / "diagnostics.json"
    diag = jsonio.load(diag_path) if diag_path.exists() else {}
    issues, notes = [], []
    if not diag:
        issues.append("没有 diagnostics.json：走步检验的行是空的，请先运行 python -m scenariolab.build --only diagnostics")
    meta = getattr(ctx, "meta", None) or {}
    P = T.month_start(ctx.panel)
    P0 = research_panel(P)
    spec = ctx.spec
    common = _common_spec(spec)
    expected = ModelSpec(factors=("oil", "ip", "cpi_goods", "cpi_services", "ff", "y10", "spread", "usd", "unrate",
                                  "gz", "tp10_x_core4"),
                         asset_factors={"energy": ["metals_oil"]}, asset_start={"gold": "1982-01"})
    is_expected = (spec.to_dict() | {"name": ""}) == (expected.to_dict() | {"name": ""})
    if not is_expected:
        issues.append(f"今天的传导模型（{spec.describe()}）不是研究轮预期的最终模型（{expected.describe()}）："
                      "“当前模型”的数字按今天的模型计算")

    # ---------------- replication
    rep = replicate(P, archive)
    if not rep["checks"]["ok"]:
        issues.append("复现研究轮存档没有完全通过：" + "；".join(rep["checks"]["count_mismatches"][:5] or
                                                     [f"最大差 {rep['checks']['required_max_delta_diff']:.1e}"]))

    # ---------------- the three base models
    b0 = ModelSpec(factors=B0_FACTORS)
    scores, per_asset_r2 = {}, {}
    for name, pan, sp in (("B0", P0, b0), ("B1", P, b0), ("final", P, spec)):
        r = {d: T.spec_score(pan, sp, *T.decade_window(d, pan)) for d in DECS}
        scores[name] = [r[d]["score"] for d in DECS]
        per_asset_r2[name] = {a: [r[d]["per_asset"][a] for d in DECS] for a in T.SCORE_ASSETS}
        if name == "final":
            n_t3 = r["T3"]["n_months"]

    # ---------------- data fixes
    bench0 = T.Bench(P0, b0)
    rep_sf = rep["candidates"].get("spread_fixed", {})
    sf_score = ([rep_sf["battery"]["base"]["common"][d]["score_ss"] for d in DECS] if rep_sf.get("available") else None)
    r, pl = bench0.transmission({"labor_share_rt": P["f_labor_share"]}, drop=("labor_share",))
    ls_fix = {"score": [float(pl.set_index("decade").loc[d, "score"]) for d in DECS],
              "per_asset": {a: [float(r[(r.asset == a) & (r.decade == d)]["delta"].iloc[0]) for d in DECS]
                            for a in T.SCORE_ASSETS}}
    old, new = P["x_spread_mixed"].loc["1973-01":"1985-12"], P["s_spread"].loc["1973-01":"1985-12"]
    sent_old_months = int((P0["f_sentiment"].loc["1960-01":"1977-12"].fillna(0) != 0).sum())
    sp_pa = {}
    if rep_sf.get("available"):
        b = rep_sf["battery"]["base"]
        sp_pa = {a: [b[a][d]["delta_ss"] for d in DECS] for a in rep_sf["assets"]}
    data_fixes = [
        {"id": "spread_timing", "name": "信用利差计时修正", "changes_model": True,
         "before": "1962–85 年用 Baa 月均收益率减 10 年期月末收益率", "after": "两条腿都取当月最后一周（1986 年起都取月末）",
         "score_delta": sf_score,
         "per_asset": sp_pa,
         "evidence": {"level_corr_1973_1985": float(old.corr(new)), "mean_abs_diff_1973_1985": float((new - old).abs().mean()),
                      "max_abs_diff_1973_1985": float((new - old).abs().max()),
                      "identical_from": "1986-01", "today": float(P["s_spread"].dropna().iloc[-1]),
                      "old_today": float(P["x_spread_mixed"].dropna().iloc[-1]),
                      "source": "recomputed"}},
        {"id": "release_dating", "name": "季度数据按发布月对齐", "changes_model": True,
         "before": "季度值放在季中月份并线性插值：一个月里已含下一季度的部分变化",
         "after": "每个季度值在首次发布的月份出现并保持到下一次发布（劳动收入占比滞后 4 个月，NIPA 5 个月，Z.1 6 个月）",
         "score_delta": ls_fix["score"], "per_asset": ls_fix["per_asset"],
         "evidence": {"release_lags": meta.get("release_lags") or config.RELEASE_LAGS,
                      "labor_share_nonzero_months_per_year": float((P["f_labor_share"].loc["1990-01":"1999-12"].fillna(0) != 0).sum() / 10),
                      "old_nonzero_months_per_year": float((P0["f_labor_share"].loc["1990-01":"1999-12"].fillna(0) != 0).sum() / 10),
                      "look_ahead_corr_archive": 1.0, "source": "recomputed"}},
        {"id": "sentiment_hold", "name": "1978 年前消费者信心不插值", "changes_model": False,
         "before": "1978 年前的季度调查之间线性插值", "after": "调查值保持到下一次调查（最多 3 个月），只在调查月变化",
         "score_delta": None, "per_asset": {},
         "evidence": {"nonzero_months_1960_1977_new": int((P["f_sentiment"].loc["1960-01":"1977-12"].fillna(0) != 0).sum()),
                      "nonzero_months_1960_1977_research": sent_old_months, "source": "recomputed",
                      "note": "只影响候选（加消费者信心）和以它为视图的情景"}},
    ]

    # ---------------- changes (common and dedicated)
    alone = {}
    bench1 = T.Bench(P, b0)
    for cid, cand, drop in (("add_gz", P.get("f_gz"), ()), ("add_tp10_x_core4", P.get("f_tp10_x_core4"), ()),
                            ("drop_labor_share", None, ("labor_share",))):
        if cand is None and not drop:
            continue
        _, pl = bench1.transmission({cid: cand}, drop=drop)
        alone[cid] = [float(pl.set_index("decade").loc[d, "score"]) for d in DECS]
    s_common = ModelSpec(factors=tuple(f for f in expected.factors))      # the research's S, as a fallback core
    core = common if {"gz", "tp10_x_core4"} <= set(common.factors) else s_common
    ded = {}
    for cid, change, asset in (("energy_metals_oil", {"asset_factors": {"energy": ["metals_oil"]}}, "energy"),
                               ("gold_window_1982", {"asset_start": {"gold": "1982-01"}}, "gold"),
                               ("gold_window_1985", {"asset_start": {"gold": "1985-01"}}, "gold")):
        if "f_metals_oil" not in P.columns and asset == "energy":
            continue
        res = _decade_scores(P, {"core": core, "change": _with(core, **change)})
        ded[cid] = {"asset": asset, "core": core.describe(),
                    "r2_core": [res["core"][d]["per_asset"][asset] for d in DECS],
                    "r2_change": [res["change"][d]["per_asset"][asset] for d in DECS],
                    "gain": [res["change"][d]["per_asset"][asset] - res["core"][d]["per_asset"][asset] for d in DECS],
                    "score_gain": [res["change"][d]["score"] - res["core"][d]["score"] for d in DECS]}
    loo_final = _loo(P, spec)
    loo_common = _loo(P, core)
    spread_in_final = _spread_fix_in(P, spec)
    arc_bat = archive["battery"]
    endo_gz = arc_bat["risk:gz"].get("endogeneity", {})
    split = archive["rederived"]["tp_core4_mechanism"]["A_T3_split"]
    act = rep["core4_active_months"]
    ls_loo = loo_common.get("labor_share") or loo_final.get("labor_share")     # in the common spec, as the walk-forward chose it

    def rep_ref(cid, key):
        c = rep["candidates"].get(cid, {})
        return {**_refutation_row(c["battery"], key), "source": "recomputed"} if c.get("available") else None

    def rep_block(cid):
        c = rep["candidates"].get(cid, {})
        if not c.get("available"):
            return None
        b = c["battery"]["base"]
        return {"per_asset": {a: {d: {"delta": b[a][d]["delta_ss"], "dm_t": b[a][d]["dm_t"], "cw_t": b[a][d]["cw_t"]}
                                  for d in DECS} for a in c["assets"]},
                "common": {d: {"score": b["common"][d]["score_ss"], "pooled_t": b["common"][d]["pooled_t"]} for d in DECS},
                "z": {a: v[0] for a, v in c["battery"]["z"].items()},
                "p": {a: float(1 - _norm_cdf(v[0])) for a, v in c["battery"]["z"].items()},
                "q_archive": _q_archive(archive, c["archive_key"]), "diff": c["diff"]}
    changes = [
        {"id": "spread_timing", "name": "信用利差计时修正", "kind": "data_fix", "walkforward_id": None,
         "definition": "信用利差 = Baa 公司债收益率 − 同一时点的 10 年期国债收益率；1962–85 年两条腿都取当月最后一周（FRED WBAA、WGS10YR），1986 年起都取月末（DBAA、DGS10）",
         "alignment": "月末口径（1962 年以前仍是月均）", "coverage": "1962-02 起每个月",
         "table": {"values": sf_score, "basis": "vs_B0"},
         "walkforward": None, "replication": rep_block("spread_fixed"), "refutation": rep_ref("spread_fixed", "common"),
         "loo_in_final": spread_in_final, "loo_in_common": _spread_fix_in(P, core),
         "verdict": "CONFIRMED", "status": "fixed",
         "caveats": ["月末时点的公司债收益率仍然滞后于股票（1986–2026 年同月相关 −0.40，上月 −0.27）：这是更好的度量，不是完美的度量。",
                     "旧序列保留为审计列，用来复现研究轮的基准。"],
         "caveat_sources": ["archive", None]},
        {"id": "release_dating", "name": "季度数据按发布月对齐", "kind": "data_fix", "walkforward_id": None,
         "definition": "季度值（劳动收入占比、NIPA 利润与投资、Z.1 财富）在首次发布的月份出现，保持到下一次发布；比率先在季度频率上计算",
         "alignment": "按发布月（劳动收入占比与生产率滞后 4 个月，NIPA 5 个月，Z.1 6 个月）", "coverage": "1947 年起",
         "table": {"values": ls_fix["score"], "basis": "vs_B0"},
         "walkforward": None, "replication": None, "refutation": None,
         "loo_in_final": None, "loo_in_common": None, "verdict": "CONFIRMED", "status": "fixed",
         "caveats": ["插值的劳动收入占比在每季第一个月就与本季的变化相关 1.000000，而这个变化 4–5 个月后才发布：这是前视，修正是必须的，不算增益。"],
         "caveat_sources": ["archive"]},
        {"id": "sentiment_hold", "name": "1978 年前消费者信心不插值", "kind": "data_fix", "walkforward_id": None,
         "definition": "1978 年前密歇根调查是季度的（2、5、8、11 月）：调查值保持到下一次调查，最多 3 个月",
         "alignment": "调查月", "coverage": "1952 年起", "table": {"values": None, "basis": None},
         "walkforward": None, "replication": None, "refutation": None, "loo_in_final": None, "loo_in_common": None,
         "verdict": None, "status": "fixed", "caveats": ["只影响候选和以消费者信心为视图的情景，不改变模型。"]},
        {"id": "add_gz", "name": "加 GZ 信用利差", "kind": "common", "walkforward_id": "add_gz",
         "definition": "GZ 信用利差（美联储超额债券溢价文件：高级无担保公司债相对同期限国债的平均利差，取自二级市场价格）的月度变化",
         "alignment": "参考月（美联储在下月第一周发布）", "coverage": "1973-02 起每个月",
         "table": {"values": alone.get("add_gz"), "basis": "alone_on_B1"},
         "walkforward": _walkforward(diag, "add_gz"), "replication": rep_block("gz"), "refutation": rep_ref("gz", "common"),
         "loo_in_final": loo_final.get("gz"), "loo_in_common": loo_common.get("gz"), "verdict": "CONFIRMED",
         "status": "adopted" if "gz" in spec.factors else "not_in_model",
         "caveats": [f"GZ 是公司债市场价格，当月资产收益能解释它变化的 {endo_gz.get('r2_assets', float('nan')):.0%}；"
                     "美联储每月都可能修订整段历史，2026 年 9 月起部分数据换了供应商。",
                     "模拟里它的变化取自抽中的历史月份，不按今天的利差水平缩放（信用利差变动会缩放）。"],
         "caveat_sources": ["archive", None]},
        {"id": "add_tp10_x_core4", "name": "加 期限溢价变动 × 上月核心 CPI > 4%", "kind": "common",
         "walkforward_id": "add_tp10_x_core4",
         "definition": "ACM 10 年期限溢价的月度变化，只在上月核心 CPI 同比高于 4% 的月份进入（其余月份为 0）",
         "alignment": "月末；哑变量用上月核心 CPI", "coverage": "1961-07 起每个月",
         "table": {"values": alone.get("add_tp10_x_core4"), "basis": "alone_on_B1"},
         "walkforward": _walkforward(diag, "add_tp10_x_core4"), "replication": rep_block("tp10_x_core4"),
         "refutation": rep_ref("tp10_x_core4", "common"),
         "loo_in_final": loo_final.get("tp10_x_core4"), "loo_in_common": loo_common.get("tp10_x_core4"),
         "verdict": "CONFIRMED", "status": "adopted" if "tp10_x_core4" in spec.factors else "not_in_model",
         "active_months": act,
         "caveats": [f"1995–2014 年的测试月份里核心 CPI 从未高于 4%（这两个十年的测试期里高通胀月份 "
                     f"{act['T1']['test_active'] + act['T2']['test_active']} 个，1973–2014 年的训练期里 "
                     f"{act['T3']['train_active']} 个）：这两个十年的提升来自正常时期的系数更干净；2015–26 年的提升中，"
                     f"2021–23 年 {split['n_active']} 个高通胀月份贡献 {split['from_active_months']:+.3f}"
                     f"（共 {split['score_total']:+.3f}）。ACM 期限溢价是在全样本上估计的模型值。",
                     "期限溢价项只在抽中的历史月份本身处于高核心通胀时起作用；模拟路径的通胀升高不会自动打开它。"],
         # the active-month counts are recomputed (replication), the 2021-23 split is the archive's
         "caveat_sources": ["archive", None]},
        {"id": "drop_labor_share", "name": "去掉劳动收入占比", "kind": "common", "walkforward_id": "drop_labor_share",
         "definition": "劳动收入占比（按发布月对齐）移出传导模型；它仍是模拟的宏观路径和情景变量",
         "alignment": "按发布月（滞后 4 个月）", "coverage": "1947 年起",
         "table": {"values": alone.get("drop_labor_share"), "basis": "alone_on_B1"},
         "walkforward": _walkforward(diag, "drop_labor_share"), "replication": None, "refutation": None,
         "loo_in_final": loo_final.get("labor_share"), "loo_in_common": loo_common.get("labor_share"),
         "verdict": "PLAUSIBLE", "status": "adopted" if "labor_share" not in spec.factors else "not_in_model",
         "caveats": [_labor_share_caveat(ls_loo)]},
        {"id": "energy_metals_oil", "name": "能源股 + 基础金属相对油价", "kind": "dedicated", "walkforward_id": "energy_metals_oil",
         "asset": "energy",
         "definition": "世界银行基础金属指数（不含铁矿石）除以原油均价，取对数变化；只进能源股方程",
         "alignment": "月均（两者都是月均价格）", "coverage": "1960-02 起每个月",
         "table": {"values": ded.get("energy_metals_oil", {}).get("gain"), "basis": "asset_on_core"},
         "dedicated": ded.get("energy_metals_oil"),
         "walkforward": _walkforward(diag, "energy_metals_oil"), "replication": rep_block("metals_oil"),
         "refutation": rep_ref("metals_oil", "energy"),
         "loo_in_final": loo_final.get("metals_oil"), "loo_in_common": None, "verdict": "CONFIRMED",
         "status": "adopted" if "metals_oil" in asset_extras(spec).get("energy", ()) else "not_in_model",
         "caveats": ["幅度小；一部分是油价计时（单用月均油价就有一部分增益）。",
                     "在走步检验第 1 轮只比单个资产的门槛高一点点；最终规则看最后一轮。"]},
        {"id": "gold_window", "name": "黄金从 1982 年起估计", "kind": "dedicated", "walkforward_id": "gold_window",
         "asset": "gold",
         "definition": "黄金方程只用 1982-01 以后的月份估计（另一个预先登记的选项是 1985-01）；不是新因子",
         "alignment": "—", "coverage": "—",
         "table": {"values": ded.get("gold_window_1982", {}).get("gain"), "basis": "asset_on_core"},
         "dedicated": ded.get("gold_window_1982"), "dedicated_alt": ded.get("gold_window_1985"),
         "walkforward": _walkforward(diag, "gold_window"), "replication": None, "refutation": None,
         "loo_in_final": loo_final.get("gold_window"), "loo_in_common": None, "verdict": "PLAUSIBLE",
         "status": "adopted" if dict(spec.asset_start).get("gold") else "not_in_model", "flagged": True,
         "caveats": ["这一改动是在同一批十年上发现的，走步检验对它不是独立确认；从 2026-10 起的实时记录才是。",
                     "1973–81 年包含布雷顿森林体系后的重新定价、官方售金和 1979–80 年的暴涨；在它上面估计的固定系数会扭曲之后的关系。"]},
    ]

    # ---------------- per asset
    bound = {r["asset"]: r for r in archive["critique"]["per_asset_bound"]}
    per_asset = []
    for row in archive["per_asset"]:
        a = row["asset"]
        fin_t3 = per_asset_r2["final"][a][2]
        oracle = bound[a]["oracle_AB"]
        per_asset.append({"asset": a, "name": ASSET_NAMES.get(a, a),
                          "n_best": row["n_best"], "bh_survivors": row["bh_survivors"],
                          "bh_survivor_names": row["bh_survivor_names"],
                          "r2": {k: per_asset_r2[k][a] for k in ("B0", "B1", "final")},
                          "unexplained": 1.0 - max(fin_t3, oracle), "oracle_T3": oracle, "untested": row["untested"]})
    unexplained = float(np.mean([r["unexplained"] for r in per_asset]))

    # ---------------- residuals and out of universe
    an_b0, R0 = residual_anatomy(P0, b0)
    an_fin, _ = residual_anatomy(P, spec, base_resid=R0)
    oou = out_of_universe(P, P0, common)
    u4 = archive["critique"]["out_of_universe"]["new6"]

    # ---------------- refutation table: archive rows, the replicated ones recomputed
    names_rep = {r["archive"]: r["id"] for r in REGISTERED}
    refut = []
    for row in archive["refutation_rows"]:
        rid = names_rep.get(row["candidate"])
        c = rep["candidates"].get(rid, {}) if rid else {}
        if c.get("available"):
            key = row["asset"] or "common"
            refut.append({**row, **_refutation_row(c["battery"], key), "source": "recomputed",
                          "placebo_p_archive": row["placebo_p"]})
        else:
            refut.append(row)

    # ---------------- assemble
    tc, pc = archive["counts"]["transmission"], archive["counts"]["predictive"]
    ub = archive["critique"]["upper_bound"]
    honest = [m for m in ("A_ridge", "A_lasso", "A_pcr", "B_ridge", "B_lasso", "B_pcr") if m in ub]
    # 「诚实范围（6 种拟合）」: each asset's 2015-26 R^2 over the six honest fits, penalty chosen on the 10-asset score
    common_pa = archive["critique"].get("upper_bound_common_per_asset", {})
    honest_range = {}
    for a in T.SCORE_ASSETS:
        vals = [common_pa[m][a]["T3"] for m in honest if a in common_pa.get(m, {})]
        if len(vals) == len(honest):
            honest_range[a] = [min(vals), max(vals)]
    drift = {a: v for a, v in archive["critique"].get("beta_drift", {}).items() if isinstance(v, dict)}
    eng = archive["forecast_states"]["engine"]
    db_index = getattr(getattr(ctx, "db", None), "dates", None)
    mlp_eligible = _mlp_eligible()
    protocols = protocol_alternatives(ctx.panel, diag) if diag.get("iterations") else archive["protocol_alternatives"]
    notes += [
        "同月传导用实际的同月因子变化（衡量传导质量，不是预测）；季度数据按发布月；1962–85 年利差用每月最后一周；GZ、ACM 是今天的版本（会修订、参数用全样本估计）。",
        "晚开始的序列只和同一批训练月份的基准比较；Clark–West 在持续性回归元上偏大，用安慰剂校准；12 个月目标里油价与原油资产是同一序列，只计一次；水平类预测变量要做去趋势检验。",
        "走步检验：测试期参与了“采用 / 不采用”的决定，没有未动用的十年了；从 2026-10 起记录实时预测，明年检验。",
        "第 2 轮的改动要胜过本轮起始模型（不只是基准）才保留；神经网络只作对照。",
        "宏观体制的高通胀看当月 CPI 同比；传导模型的期限溢价项看上月核心 CPI 同比；门槛都是 4%。",
        "结构冲击是在全样本上识别的（验证期和测试期对这些列存在轻微前视）：本轮修正利差计时后它们会重新识别，第 2 轮的数字随之变化，决定不变。",
    ]
    out = {
        "version": VERSION,
        "data_version": {"panel_sha1": _sha(config.PANEL_PATH),
                         "model_selection_sha1": _sha(config.RESULTS_DIR / "model_selection.json"),
                         "diagnostics_sha1": _sha(diag_path), "data_end": T.ym(P.index.max())},
        "archive": {"id": archive["id"], "access_date": archive["access_date"], "panel_sha1": archive["panel_sha1"],
                    "tag": archive["tag"], "docs": archive["docs"], "source": "archive"},
        "model": {"spec": spec.to_dict(), "describe": spec.describe(), "is_expected": bool(is_expected),
                  "expected_describe": expected.describe(), "common_describe": common.describe()},
        "summary": {
            "n_transmission": tc["n"], "n_predictive": pc["n"], "families": archive["counts"]["families"],
            "transmission": {k: tc["verdicts"].get(k, 0) for k in ("CONFIRMED", "PLAUSIBLE", "REFUTED")},
            "predictive": {k: pc["verdicts"].get(k, 0) for k in ("PLAUSIBLE", "REFUTED")},
            "changes": {"data_fixes": sum(c["kind"] == "data_fix" for c in changes),
                        "data_fixes_changing_model": sum(f["changes_model"] for f in data_fixes),
                        "common": sum(c["kind"] == "common" for c in changes),
                        "dedicated": sum(c["kind"] == "dedicated" for c in changes)},
            "adopted_in_model": [c["id"] for c in changes if c.get("status") == "adopted"],
            "score": scores, "n_test_months_T3": n_t3,
            "unexplained_share_2015_2026": unexplained,
            "honest_bound_2015_2026": [0.18, 0.23],
            "honest_fits_T3": {m: ub[m]["common"]["T3"] for m in honest},
            "oracle": float(np.mean([bound[a]["oracle_AB"] for a in T.SCORE_ASSETS])),
            "fixed_core_T3": float(np.mean([bound[a]["core_fixed_T3"] for a in T.SCORE_ASSETS])),
            "useful_set_T3": float(np.mean([bound[a]["useful_U5_T3"] for a in T.SCORE_ASSETS])),
            "replication_ok": rep["checks"]["ok"],
            # the summary mixes both: counts and bounds are the archive's, the scores this build's; the
            # unexplained share is 1 - max(this build's final model, the archive's oracle penalty) per asset
            "sources": {"counts": "archive", "score": "recomputed", "honest_bound_2015_2026": "archive",
                        "honest_fits_T3": "archive", "oracle": "archive", "fixed_core_T3": "archive",
                        "useful_set_T3": "archive", "unexplained_share_2015_2026": "mixed"}},
        "coverage": coverage(P, meta),
        "data_fixes": data_fixes,
        "changes": changes,
        "plausible": [{**p, "source": "archive"} for p in archive["plausible"]],
        "families": [{**f, "source": "archive"} for f in archive["families"]],
        "per_asset": per_asset,
        "bh": {"transmission_pairs": {k: archive["bh"]["transmission_pairs"][k] for k in ("n", "rejections", "rows", "statistic")},
               "score": {k: archive["bh"]["score"][k] for k in ("n", "rejections", "rows", "top10")},
               "raw_dm_would_add": archive["bh"]["raw_dm_would_add"],
               "prediction": {k: archive["bh"]["prediction"][k] for k in ("n", "cw", "dm", "calibrated", "rows",
                                                                          "calibrated_rows")},
               "cw_size_on_placebos": archive["bh"]["cw_size_on_placebos"], "source": "archive"},
        "refutation": refut,
        "forecast_states": {
            "replicated": rep["predictive"], "plausible": archive["forecast_states"]["rows"],
            "engine": {k: _engine_row(eng.get(k)) for k in ("base", "re_loans_yoy", "labor_share_rt", "china_import_share")},
            "engine_reproduces_build": eng.get("base_reproduces_build"), "source": "mixed"},
        "upper_bound": {"per_asset": [{**r, "honest_range_T3": honest_range.get(r.get("asset"))}
                                      for r in archive["critique"]["per_asset_bound"]],
                        "honest_fits": honest,
                        "beta_drift": {"by_decade_adj_r2": {a: v["by_decade_adj_r2"] for a, v in drift.items()},
                                       "constant_adj_r2": {a: v["constant_adj_r2"] for a, v in drift.items()},
                                       "by_decade_range": ([min(v["by_decade_adj_r2"] for v in drift.values()),
                                                            max(v["by_decade_adj_r2"] for v in drift.values())]
                                                           if drift else None)},
                        "methods": {m: {**ub[m]["common"], "n_candidates": ub[m].get("n_candidates")}
                                    for m in ub if isinstance(ub[m], dict) and "common" in ub[m]},
                        "residual_anatomy": archive["critique"]["residual_anatomy"], "source": "archive"},
        "residuals": {"B0": an_b0, "final": an_fin, "final_describe": spec.describe(), "source": "recomputed",
                      "fallback": _residual_fallback(spec), "note": _residual_note(spec)},
        "model_vs_factor": {"model_on_base": archive["critique"]["model_vs_factor"]["model_on_base"],
                            "model_on_U4": archive["critique"]["model_vs_factor"]["model_on_U4"],
                            "factor_on_base": archive["critique"]["model_vs_factor"]["factor_on_base"],
                            "mlp_seeds": archive["critique"]["mlp_seeds"], "source": "archive"},
        "out_of_universe": {"archive_U4": {"delta_vs_base": [u4["delta_vs_base"]["U4 + tp10_x_infl4"][d] for d in DECS],
                                           "n_better": [u4["n_better_U4"][d] for d in DECS], "source": "archive"},
                            "final_common": {**oou, "source": "recomputed"} if oou else None},
        "protocol": {"gate": "incremental", "mlp_eligible": mlp_eligible,
                     "alternatives": protocols, "source": "recomputed" if protocols else "archive",
                     "archive_alternatives": archive["protocol_alternatives"],
                     "note": "研究轮用的是修正利差计时之前识别的结构冲击；本次构建用重新识别的结构冲击重算了这几种协议，"
                             "第 2 轮的数字因此和研究轮存档略有不同"},
        "concepts": [{**c, "columns_present": [col for col in c["new_columns"] if col in P.columns], "source": "archive"}
                     for c in archive["concepts"]],
        "concept_counts": _status_counts(archive["concepts"]),
        "today": today(P, db_index),
        "replication": {"checks": rep["checks"], "candidates": {k: {kk: v.get(kk) for kk in ("name", "archive_key", "assets",
                                                                                               "diff", "z_archive", "available")}
                                                                for k, v in rep["candidates"].items()}},
        "untested": ["CFNAI / ADS 实时经济活动指数", "医疗 CPI 与药品 PPI（医疗保健股）", "采暖与制冷度日（公用事业股）",
                     "Z.1 商业地产价格（REITs）", "券商杠杆与净派息（预测方面）", "德国与日本 10 年期国债",
                     "粘性 / 中位数 CPI（CPI 预测）", "密歇根失业预期", "核心宏观数据的首发值（费城联储 RTDSM）"],
        "no_source": [c["name"] for c in archive["concepts"] if c["status"] == "none"] + ["月末金价"],
        "issues": issues, "notes": notes,
    }
    return out


# the comparison protocols of FACTORS_DESIGN §2.3, rerun on this build's panel and structural shocks (stage A only:
# stage B acts on single assets and does not change the comparison)
PROTOCOLS = [
    {"id": "old_gate", "name": "神经网络只作对照、旧关卡（只和基准比）", "gate": "base", "mlp_eligible": False},
    {"id": "mlp_eligible", "name": "神经网络可被选中、增量关卡", "gate": "incremental", "mlp_eligible": True},
    {"id": "mlp_eligible_old_gate", "name": "神经网络可被选中、旧关卡", "gate": "base", "mlp_eligible": True},
]


def _protocol_row(pid: str, name: str, res: dict, gate: str, derived: bool = False) -> dict:
    """One protocol's iteration-2 selection, its test against the start and the base, and the common spec it
    keeps. `derived`: an old-gate row read off a two-iteration incremental run of the same candidates. The gate
    changes only the last decision there: iteration 1 starts from the base under either gate (its test against
    the start is its test against the base), so iteration 2 starts from the same spec."""
    from . import walkforward as wf
    last = res["iterations"][-1]
    tr = last["test_result"]
    if derived:
        sel = ModelSpec.from_dict(last["selected"]["spec"])
        improved = bool(not wf.same_spec(sel, ModelSpec()) and tr["delta"] > wf.MIN_GAIN)
        final, kept = (last["selected"]["describe"], "selected") if improved else (ModelSpec().describe(), "base")
    else:
        common = res["final"]["common"]
        improved, final, kept = bool(tr["improved"]), common["describe"], common.get("kept")
    return {"id": pid, "name": name, "gate": gate, "iteration2": list(last["selected"]["changes"]), "final": final,
            "kept": kept, "t3_vs_start": tr.get("delta_vs_start"), "t3_vs_base": tr.get("delta"),
            "t3_selected": tr["selected"]["score"], "t3_start": tr["start"]["score"], "n_months": tr["n_months"],
            "improved": improved}


def protocol_alternatives(panel: pd.DataFrame, diag: dict, protocols: list[dict] | None = None) -> list[dict]:
    """This spec's protocol (from diagnostics.json) and the alternatives of §2.3 on this build's panel
    (structural shocks included): walkforward.run_walkforward once per candidate set (the networks report-only
    or eligible) with the incremental gate; with two iterations the old gate's row is read off the same run."""
    from . import walkforward as wf
    rows, runs = [], {}
    if diag.get("iterations"):
        rows.append(_protocol_row("spec", "本轮协议（神经网络只作对照、增量关卡）", diag, "incremental"))
        runs[bool(diag.get("mlp_eligible", wf.MLP_ELIGIBLE))] = diag
    for pr in PROTOCOLS if protocols is None else protocols:
        res = runs.get(pr["mlp_eligible"])
        if res is None:
            res = runs[pr["mlp_eligible"]] = wf.run_walkforward(panel, ModelSpec(), dedicated=[], gate="incremental",
                                                                mlp_eligible=pr["mlp_eligible"])
        if pr["gate"] == "incremental":
            rows.append(_protocol_row(pr["id"], pr["name"], res, pr["gate"]))
        elif len(res["iterations"]) == 2:
            rows.append(_protocol_row(pr["id"], pr["name"], res, pr["gate"], derived=True))
        else:
            other = wf.run_walkforward(panel, ModelSpec(), dedicated=[], gate=pr["gate"], mlp_eligible=pr["mlp_eligible"])
            rows.append(_protocol_row(pr["id"], pr["name"], other, pr["gate"]))
    return rows


def _residual_fallback(spec: ModelSpec, first: str = "1985-01", step: int = 12) -> dict:
    """{asset: [first, last] refit year whose own window (from its asset_start) is shorter than MIN_ASSET_WINDOW,
    so oos_residuals fits that asset from the common start there, as the engine does} for the assets with one."""
    out = {}
    for a, month in spec.asset_start:
        start, R, years = pd.Timestamp(month), pd.Timestamp(first), []
        while (R.year - start.year) * 12 + R.month - start.month < MIN_ASSET_WINDOW:
            years.append(R.year)
            R += pd.DateOffset(months=step)
        if years:
            out[a] = [years[0], years[-1]]
    return out


def _residual_note(spec: ModelSpec) -> str:
    """The residual anatomy's reading of the short-window rule (the design's d05 fitted the later starts always)."""
    fb = _residual_fallback(spec)
    if not fb:
        return "每年 1 月只用之前的数据重新估计。"
    starts = dict(spec.asset_start)
    parts = [f"{ASSET_NAMES.get(a, a)}方程（从 {starts[a][:4]} 年起估计）在 {y0}–{y1} 年的重新估计里用共同起点"
             for a, (y0, y1) in fb.items()]
    return (f"每年 1 月只用之前的数据重新估计；自己起始月份的窗口不足 {MIN_ASSET_WINDOW} 个月时，和引擎一样用共同起点："
            + "；".join(parts) + "。")


def _labor_share_caveat(loo) -> str:
    """The labor-share caveat of §6.2 C with the leave-one-out numbers of the build."""
    if not loo:
        return "按发布月对齐后，保留与否差别很小，走步检验选择去掉。它仍是情景变量和模拟里的一条路径。"
    t1, t2, t3 = loo
    tests = (f"在两个测试十年各降低 {abs(t2):.3f}、{abs(t3):.3f}" if t2 < 0 and t3 < 0
             else f"在两个测试十年分别为 {_f(t2)}、{_f(t3)}")
    return (f"按发布月对齐后，去掉它在 1995–2004 年{'提高' if t1 >= 0 else '降低'} {abs(t1):.3f}，{tests}："
            "保留与否差别很小，走步检验选择去掉。它仍是情景变量和模拟里的一条路径。")


def _mlp_eligible() -> bool | None:
    """walkforward.MLP_ELIGIBLE (the one-line switch of decision D1); True before it existed."""
    try:
        from . import walkforward
    except Exception:  # noqa: BLE001 - the audit reports, it does not need the walk-forward module
        return None
    return bool(getattr(walkforward, "MLP_ELIGIBLE", True))


def _status_counts(concepts) -> dict:
    out = {}
    for c in concepts:
        out[c["status"]] = out.get(c["status"], 0) + 1
    return out


def _engine_row(e) -> dict | None:
    if not e:
        return None
    grid = {g["ens"]: g for g in e.get("grid", [])}
    ch = e.get("chosen_ens")
    row = {"chosen_ens": ch, "risk_crps": e["all"]["risk_crps_analog"], "rank_ic": e["all"]["rank_ic_analog"],
           "cpi_crps": e["all"]["macro_crps_cpi_yoy"], "test_crps_chosen": grid.get(ch, {}).get("test_crps_analog")}
    pv = e.get("paired_vs_base_same_ens")
    if pv:
        row["paired_576"] = {k: pv[k] for k in ("risk_crps", "rank_ic", "macro_crps_cpi_yoy") if k in pv}
    return row


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _q_archive(archive: dict, key: str) -> dict:
    """The archive's BH q values of a candidate: per asset (candidate x asset family) and for the score."""
    out = {r["asset"]: r["q"] for r in archive["bh"]["transmission_pairs"]["rows"] if r["candidate"] == key}
    for r in archive["bh"]["score"]["top10"]:
        if r["candidate"] == key:
            out["score"] = r["q"]
    return out


def main() -> None:
    from ..context import EngineContext
    out = run(EngineContext.load(selection=True))
    jsonio.dump(out, config.RESULTS_DIR / "factor_audit.json")
    c = out["replication"]["checks"]
    print(f"factor_audit: replication {'ok' if c['ok'] else 'FAILED'} (max delta diff {c['required_max_delta_diff']:.1e}, "
          f"max z diff {c['required_max_z_diff']:.1e}); score B0 / B1 / final T3 "
          f"{out['summary']['score']['B0'][2]:.3f} / {out['summary']['score']['B1'][2]:.3f} / "
          f"{out['summary']['score']['final'][2]:.3f}")


if __name__ == "__main__":
    main()
