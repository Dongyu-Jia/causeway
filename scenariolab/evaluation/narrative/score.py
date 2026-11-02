"""The simulation and scoring phase of the narrative test (SIM_DESIGN §8.3, §8.6). It reads the sealed key.

For every episode of a run (after runner.translate_run): the quarterly stack is fitted on the quarters up to the
episode's train end with the train-end anchors (§5.5) and run from the origin over the window, once per arm and CMA
arm, with common random numbers (the same seed for every arm of an episode):

    U                     no statement (the model's unconditional simulation)
    O-A, O-A′, O-B, O-full  the realized x fields of the window as hard statements, by endogeneity class (the card's
                          E0 + E0p raw policy + x_oil_cause; + E1c; + E1; + E2)
    V                     O-full with the realized CAPE multiple: us_equity = realized y_mult + simulated y_fund
                          (恒等式上界，不是模型能力; us_equity only)
    N1, N2, N3, N-wrong   the run's translations (QTable/1, test mode): qtable.constraints re-reads each clean table
                          with the training statistics of THIS train end (size bins in training sd; R-RANGE against
                          the training range, policy shocks against ±4 shock sd); fields the factor layer does not
                          carry at this train end are dropped with a note; quarters beyond the window are cut; N3 is
                          re-read with numbers=False (§8.4: E1 / E1c / E2 only as size bins; the research round's
                          stand-in numbers are re-expressed as the nearest bin, and the count is recorded)
    N-synth               synthetic episodes: truth is the simulated path the narrative was written from (key.json)
and, when the run asks for engine E (engines.py), the monthly engine on the identical tables (U, the oracle arms, the
narratives): E converts exactly the clean table the Q arm ran at this train end (the same refusals, clips and
truncation; e_table), without c_q.anchors, which this version of the quarterly stack records but does not apply (so
neither engine applies them). Both arms record the table's digest (table_sha256). Each run is scored per asset and horizon (stats.score_draws). The stated and completed field values are
kept for the delivery and contamination statistics (fields.csv).

Outputs in the run directory: score_rows.csv (every cell), fields.csv (stated and completed values against realized),
episodes.json (per episode: window, arms, notes, quarterly small multiples), stats.json (pooled, paired, strata,
claims C1–C4 from the T-N record, decomposition, discrimination, chance count, MDE, n per horizon, T-S inputs).
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from ...qmodel import config_q as cq
from . import common as C
from . import key as K
from . import runner as RN
from . import stats as ST

TN_FILE = "2026-11-02_TN.json"
L0_FIELDS = tuple(cq.L0_FIELDS)
PAIRS_VS_U = ("N1", "N2", "N3", "NW", "OA", "OAp", "OB", "OF")
EXTRA_PAIRS = (("N2", "NW"), ("N1", "NW"), ("OB", "OAp"), ("N2", "N1"), ("N3", "N1"), ("N1", "OF"), ("N2", "OF"),
               ("N3", "OF"), ("OAp", "OA"), ("OF", "OB"))
QE_ARMS = ("U", "OA", "OAp", "OB", "OF", "N1", "N2", "N3", "NW")


def load_tn() -> tuple[dict, str]:
    text = (C.PREREG_DIR / TN_FILE).read_text(encoding="utf-8")
    return json.loads(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


def variant(engine: str, arm: str) -> str:
    return f"{engine}:{arm}"


# ================================================================ inputs from the panel

def window_quarters(origin: str, H: int) -> list:
    q0 = pd.Period(origin, "Q")
    return [q0 + k for k in range(int(H))]


def realized_assets(Q: pd.DataFrame, quarters) -> np.ndarray:
    """(H, 13) realized quarterly log price returns (cash: total) in config_q.ASSETS order."""
    return np.column_stack([Q[f"y_{a}"].reindex(quarters).to_numpy(float) for a in C.ASSETS])


def card_classes() -> dict[str, str]:
    from ...qmodel import card as qcard
    return {r["id"]: r["endo_code"] for r in qcard.rows(qcard.load_card(), "x")}


def oracle_values(Q: pd.DataFrame, quarters, classes, x_fields, endo: dict | None = None) -> dict[int, dict]:
    """{q (1-based): {field: realized value}} of the x fields in the arm's endogeneity classes (x_oil_cause with O-A)."""
    endo = endo or card_classes()
    want = [f for f in x_fields if endo.get(f) in classes or (f == "x_oil_cause" and "E0" in classes)]
    out = {}
    for k, q in enumerate(quarters):
        row = {}
        for f in want:
            if f in Q.columns and q in Q.index:
                v = Q.at[q, f]
                if pd.notna(v) and np.isfinite(float(v)):
                    row[f] = float(v)
        out[k + 1] = row
    return out


def oracle_statements(values: dict[int, dict]):
    from ...qmodel import factors as F
    return [F.Statement(f, (q,), v, label="oracle") for q, row in values.items() for f, v in row.items()]


def oracle_qtable(values: dict[int, dict], H: int) -> dict:
    """The same realized values as a clean QTable (the monthly engine reads it through to_phase_spec)."""
    rows = []
    for q in range(1, int(H) + 1):
        x = {}
        for f, v in (values.get(q) or {}).items():
            if f == "x_oil_cause":
                x[f] = {"value": "supply" if v >= 0.75 else ("demand" if v <= 0.25 else "mixed"), "src": "stated",
                        "quote": ""}
            else:
                x[f] = {"value": float(v), "src": "stated", "quote": ""}
        rows.append({"q": q, "x": x, "fc": {"stance": "未说明"}})
    return {"schema": cq.QTABLE_SCHEMA_ID, "model_card": cq.CARD_ID, "n_quarters": int(H), "rows": rows, "c_q": {},
            "events": [], "notes": []}


ANCHORS_NOTE = "情景的长期锚只作记录：这一版季度模型不用它们，所以月度引擎的换算也不用（两个引擎读同一张表）"


def e_table(clean: dict) -> tuple[dict, list[str]]:
    """The table the monthly engine converts for a narrative arm: the clean table the Q arm ran at this train end, without
    c_q.anchors (Q records them, W-ANCHOR, and does not apply them; to_phase_spec would turn r_star and a_tp into phase
    overrides). Returns (table, notes)."""
    cqx = dict(clean.get("c_q") or {})
    notes = []
    if cqx.pop("anchors", None):
        notes.append(ANCHORS_NOTE)
    return {**clean, "c_q": cqx}, notes


def truncate(qt: dict, H: int, x_fields) -> tuple[dict, list[str]]:
    """The clean QTable cut to the window's H quarters, without fields the factor layer does not carry."""
    notes = []
    have = set(x_fields)
    Hq = int(qt.get("n_quarters") or len(qt.get("rows") or []))
    keep_h = min(Hq, int(H))
    if Hq > H:
        notes.append(f"情景表写了 {Hq} 个季度，窗口只有 {H} 个：第 {H + 1}–{Hq} 季没有用")
    rows = []
    for r in qt.get("rows") or []:
        if int(r["q"]) > keep_h:
            continue
        x = {}
        for f, e in (r.get("x") or {}).items():
            if f in have:
                x[f] = e
            else:
                notes.append(f"第 {r['q']} 季 · {f}：这个训练期没有估计这个字段，未作输入")
        row = {"q": int(r["q"]), "x": x, "fc": r.get("fc") or {"stance": "未说明"}}
        if r.get("u"):
            row["u"] = r["u"]
        rows.append(row)
    cqx = {}
    for kname, items in (qt.get("c_q") or {}).items():
        if kname == "anchors":
            cqx[kname] = {a: v for a, v in items.items() if int(v.get("from_q", 1)) <= keep_h}
            continue
        kept = []
        for it in items or []:
            qs = [int(q) for q in it.get("quarters") or []]
            if qs and max(qs) <= keep_h:
                kept.append(it)
            elif qs:
                notes.append(f"c_q.{kname}：跨到窗口以外的陈述（第 {min(qs)}–{max(qs)} 季）没有用")
        if kept:
            cqx[kname] = kept
    out = {**qt, "n_quarters": keep_h, "rows": rows, "c_q": cqx}
    return out, notes


# ================================================================ the context check (§8.4 check_context)

def start_known(Q: pd.DataFrame, P: pd.DataFrame, origin: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The panels as they stood at the start of the origin quarter: quarterly rows up to the origin, the origin row
    keeping only what is known when the quarter starts (the states s_*, the release-lagged __known columns and the
    previous quarter's m1_* returns), and the monthly rows before the origin."""
    oq = pd.Period(origin, "Q")
    Qt = Q.loc[:oq].copy()
    keep = [c for c in Qt.columns if c.startswith("s_") or "__known" in c or c.startswith("m1_")]
    if oq in Qt.index:
        Qt.loc[oq, [c for c in Qt.columns if c not in keep]] = np.nan
    Pt = P.loc[:oq.start_time - pd.Timedelta(days=1)]
    return Qt, Pt


def check_context(train_end: str, origin: str, *, Q: pd.DataFrame | None = None, P: pd.DataFrame | None = None,
                  H: int = 4, n_paths: int = 200, seed: int = 0, n_boot: int = 4) -> dict:
    """§8.4 check_context: an episode's stack fitted on the full panels (what score_episode passes to fit_stack) and on
    the panels cut at the origin (start_known) give the same no-narrative (U) paths for a fixed seed. Returns
    {"max_abs_diff": {"price", <factor fields>}, "ok"}: nothing after T0 reaches the episode's context."""
    from ...data import panel as mpanel
    from ...qmodel import panel as qpanel
    from ...qmodel import rollout as R
    Q = qpanel.load_panel() if Q is None else Q
    P = mpanel.load_panel() if P is None else P
    Qt, Pt = start_known(Q, P, origin)
    out = {}
    rolls = []
    for q_, p_ in ((Q, P), (Qt, Pt)):
        # an episode's origin is in the past: both fits read only what was published by its start (the truncated
        # panel ends at the origin, where 'auto' would read it as today's row and take every value it holds)
        st = R.fit_stack(train_end, origin=origin, anchors="train_end", seed=0, n_boot=n_boot, Q=q_, P=p_,
                         information="published")
        rolls.append(R.run_path(st, [], [], H=H, n_paths=n_paths, seed=seed, cma="historical", states=False,
                                mode="test"))
    a, b = rolls
    out["price"] = float(np.nanmax(np.abs(a.ax.price - b.ax.price)))
    for f in ("d_ff", "d_unrate", "d_gz", "d_cpi_goods", "d_oil"):
        if f in a.fx.x and f in b.fx.x:
            out[f] = float(np.nanmax(np.abs(np.asarray(a.fx.x[f]) - np.asarray(b.fx.x[f]))))
    return {"max_abs_diff": out, "ok": all(v <= 1e-9 for v in out.values()), "train_end": train_end, "origin": origin}


# ================================================================ one episode

class Episode:
    """One episode's fitted stack, window and realized values, and the arms run on it."""

    def __init__(self, code: str, entry: dict, *, Q: pd.DataFrame, seed: int, n_paths: int, n_boot: int | None,
                 cmas=C.CMA_ARMS):
        from ...qmodel import panel as qpanel
        from ...qmodel import rollout as R
        self.code, self.entry, self.Q = code, entry, Q
        self.synth = entry.get("kind") == "synth"
        self.H = int(entry["n_quarters"])
        self.train_end = entry["train_end"]
        self.origin = entry["origin"]
        self.seed, self.n_paths, self.cmas = int(seed), int(n_paths), tuple(cmas)
        self.quarters = window_quarters(self.origin, self.H)
        t0 = time.perf_counter()
        self.stack = R.fit_stack(self.train_end, origin=self.origin, anchors="train_end", seed=0, n_boot=n_boot, Q=Q,
                                 information="published")
        self.fit_s = round(time.perf_counter() - t0, 2)
        if self.synth:
            self.realized = np.asarray(entry["truth"]["assets"], float)
            self.realized_x = {f: np.asarray(v, float) for f, v in entry["truth"]["x"].items()}
        else:
            self.realized = realized_assets(Q, self.quarters)
            self.realized_x = {f: Q[f].reindex(self.quarters).to_numpy(float) for f in L0_FIELDS if f in Q.columns}
        fs = qpanel.field_stats(Q, [f for f in self.stack.x_fields if f in Q.columns and f != "x_oil_cause"],
                                self.train_end)
        self.sd = {f: float(fs.at[f, "sd"]) for f in fs.index}
        self.mean = {f: float(fs.at[f, "mean"]) for f in fs.index}
        self.rolls: dict = {}
        self.notes: dict = {}

    def run(self, arm: str, statements=(), valuation=(), cycle_odds=None) -> dict:
        """{cma: Rollout} of one arm (states are not rolled: scoring needs the returns and the factor paths)."""
        from ...qmodel import rollout as R
        out = {}
        for cma in self.cmas:
            out[cma] = R.run_path(self.stack, list(statements), list(valuation), H=self.H, n_paths=self.n_paths,
                                  seed=self.seed, cma=cma, cycle_odds=cycle_odds, states=False, mode="test")
        self.rolls[arm] = out
        return out

    def rows(self, arm: str, engine: str = "Q", paths: np.ndarray | None = None, cma: str | None = None) -> list[dict]:
        base = {"code": self.code, "set": self.entry["set"], "bust": self.entry.get("bust"), "engine": engine,
                "arm": arm, "variant": variant(engine, arm)}
        out = []
        if paths is not None:
            for r in ST.score_draws(paths, self.realized, self.H):
                out.append({**base, "cma": cma or "historical", **r})
            return out
        for c, roll in self.rolls[arm].items():
            for r in ST.score_draws(roll.ax.price, self.realized, self.H):
                out.append({**base, "cma": c, **r})
        return out


def _small_multiple(paths: np.ndarray, realized: np.ndarray, H: int) -> dict:
    cum = np.cumsum(np.asarray(paths, float)[:, :H, :], axis=1)
    yc = np.cumsum(realized[:H], axis=0)
    out = {}
    for a in C.SMALL_MULTIPLE_ASSETS:
        j = C.ASSETS.index(a)
        q = np.nanpercentile(cum[:, :, j], [10, 50, 90], axis=0)
        out[a] = {"p10": np.round(q[0], 4).tolist(), "p50": np.round(q[1], 4).tolist(),
                  "p90": np.round(q[2], 4).tolist(), "realized": np.round(yc[:, j], 4).tolist()}
    return out


def _entry_value(e: dict, sd: float | None) -> float | None:
    if "value" in e and isinstance(e["value"], (int, float)):
        return float(e["value"])
    if "sign" in e and sd:
        if e["sign"] == "0":
            return 0.0
        return (1.0 if e["sign"] == "+" else -1.0) * cq.SIZE_BINS.get(e.get("size") or "小", 0.25) * sd
    return None


def field_rows(ep: Episode, arm: str, qt: dict, cma: str) -> list[dict]:
    """Stated values (and the completed L0 fields) of one translated arm against realized, in training sd."""
    roll = ep.rolls[arm][cma]
    u = ep.rolls["U"][cma] if "U" in ep.rolls else None
    out = []
    stated_q: dict[int, set] = {}
    for r in qt.get("rows") or []:
        q = int(r["q"])
        if q > ep.H:
            continue
        for f, e in (r.get("x") or {}).items():
            if f == "x_oil_cause" or f not in roll.fx.x:
                continue
            stated_q.setdefault(q, set()).add(f)
            v = _entry_value(e, ep.sd.get(f))
            if v is None:
                continue
            real = ep.realized_x.get(f)
            out.append({"code": ep.code, "arm": arm, "field": f, "q": q, "stated": v, "src": e.get("src"),
                        "form": "value" if "value" in e else "sign", "completed": False,
                        "realized": float(real[q - 1]) if real is not None else np.nan,
                        "sd": ep.sd.get(f, np.nan), "prior_mean": ep.mean.get(f, np.nan),
                        "sim_q50": float(np.nanmedian(roll.fx.x[f][:, q - 1])),
                        "u_q50": float(np.nanmedian(u.fx.x[f][:, q - 1])) if u is not None else np.nan})
    for f in L0_FIELDS:
        if f not in roll.fx.x or f not in ep.realized_x:
            continue
        for q in range(1, ep.H + 1):
            if f in stated_q.get(q, set()):
                continue
            out.append({"code": ep.code, "arm": arm, "field": f, "q": q, "stated": np.nan, "src": None, "form": None,
                        "completed": True, "realized": float(ep.realized_x[f][q - 1]), "sd": ep.sd.get(f, np.nan),
                        "prior_mean": ep.mean.get(f, np.nan), "sim_q50": float(np.nanmedian(roll.fx.x[f][:, q - 1])),
                        "u_q50": float(np.nanmedian(u.fx.x[f][:, q - 1])) if u is not None else np.nan})
    return out


def score_episode(code: str, entry: dict, trans: dict, *, Q: pd.DataFrame, seed: int, n_paths: int,
                  n_boot: int | None, engines=("Q",), e_paths: int = C.E_PATHS, cmas=C.CMA_ARMS,
                  oracle_arms=C.ORACLE_ARMS, e_arms=None) -> dict:
    """{"rows", "fields", "summary"} of one episode. trans: {arm: translation record} for this code."""
    from ...qmodel import qtable as QT
    summary = {"code": code, "set": entry["set"], "kind": entry.get("kind", "episode"), "n_quarters": entry["n_quarters"],
               "arms": {}, "translations": {}, "notes": [], "skipped": False}
    try:
        ep = Episode(code, entry, Q=Q, seed=seed, n_paths=n_paths, n_boot=n_boot, cmas=cmas)
    except Exception as err:  # noqa: BLE001 - an episode the stack cannot fit is reported, not fatal
        summary.update(skipped=True, reason=f"季度模型在这个训练期拟合不出来：{type(err).__name__}: {err}")
        return {"rows": [], "fields": [], "summary": summary}
    summary.update(stack=ep.stack.version, fit_seconds=ep.fit_s, origin=ep.origin, train_end=ep.train_end)
    rows, fields, series = [], [], {}
    head = cmas[0]
    endo = card_classes()

    def record(arm, rolls, notes=None, extra=None):
        roll = rolls[head]
        summary["arms"][arm] = {"notes": (notes or []) + [n.get("text") if isinstance(n, dict) else str(n)
                                                          for n in (roll.notes or [])][:12],
                                "statements": len(roll.statements), "tail_share": roll.info.get("tail_share"),
                                **(extra or {})}
        series[variant("Q", arm)] = _small_multiple(roll.ax.price, ep.realized, ep.H)
        rows.extend(ep.rows(arm))

    record("U", ep.run("U"))
    oracle_vals = {}
    if not ep.synth:
        for arm in oracle_arms:
            vals = oracle_values(Q, ep.quarters, C.ARMS[arm]["classes"], ep.stack.x_fields, endo)
            oracle_vals[arm] = vals
            try:
                record(arm, ep.run(arm, oracle_statements(vals)),
                       extra={"stated_fields": sorted({f for r in vals.values() for f in r})})
            except Exception as err:  # noqa: BLE001
                summary["arms"][arm] = {"skipped": True, "reason": f"{type(err).__name__}: {err}"}
        if "OF" in ep.rolls:                    # V: the identity bound on us_equity
            j = C.ASSETS.index("us_equity")
            ym = Q["y_mult"].reindex(ep.quarters).to_numpy(float)
            if np.isfinite(ym).all():
                for c, roll in ep.rolls["OF"].items():
                    p = np.zeros_like(roll.ax.price)
                    p[:, :, j] = ym[None, :] + roll.ax.y_fund
                    for r in ST.score_draws(p, ep.realized, ep.H):
                        if r["asset"] == "us_equity":
                            rows.append({"code": code, "set": entry["set"], "bust": entry.get("bust"), "engine": "Q",
                                         "arm": "V", "variant": variant("Q", "V"), "cma": c, **r})
                summary["arms"]["V"] = {"notes": [C.LABELS["identity"]], "assets": ["us_equity"]}
    else:
        vals = {k + 1: {f: float(v[k]) for f, v in ep.realized_x.items() if np.isfinite(v[k])} for k in range(ep.H)}
        oracle_vals["OF"] = vals
        record("OF", ep.run("OF", oracle_statements(vals)), notes=["合成片段：已知的是模拟路径自己的因子"])
    clean_tables: dict[str, dict] = {}        # the table each translated arm ran on Q (E converts the same one)
    for arm, rec in sorted(trans.items()):
        qt0 = rec.get("qtable")
        if not isinstance(qt0, dict):
            summary["translations"][arm] = {"status": "missing"}
            continue
        qt, notes = truncate(qt0, ep.H, ep.stack.x_fields)
        tinfo = {"status": rec.get("status"), "label": rec.get("label"), "source_kind": rec.get("source_kind"),
                 "refused": len((rec.get("validation") or {}).get("refused") or []),
                 "missing": [m.get("rule") for m in (rec.get("validation") or {}).get("missing") or []],
                 "entries": ((rec.get("validation") or {}).get("counts") or {}).get("entries"),
                 "narrative": rec.get("source")}
        try:
            read = QT.constraints(qt, stack=ep.stack, mode="test", narrative=None,
                                  source="legacy" if rec.get("source_kind") == "legacy" else "translator", strict=False,
                                  numbers=arm != "N3")
            clean = read["report"]["qtable"]
            sha = QT.digest(clean)
            rolls = ep.run(arm, read["statements"], read["valuation"], read.get("cycle_odds"))
            n_binned = len(read["report"].get("binned") or [])
            record(arm, rolls, notes=notes + [n["text"] for n in read["notes"]][:10],
                   extra={"refused_at_train_end": len(read["report"]["refused"]), "table_sha256": sha,
                          "n3_binned": n_binned})
            clean_tables[arm] = clean
            fields.extend(field_rows(ep, arm, clean, head))
            tinfo["refused_at_train_end"] = len(read["report"]["refused"])
            tinfo["n3_binned"] = n_binned
            tinfo["table_sha256"] = sha
        except Exception as err:  # noqa: BLE001 - one bad table does not stop the episode
            summary["arms"][arm] = {"skipped": True, "reason": f"{type(err).__name__}: {err}"}
        summary["translations"][arm] = tinfo
    if "E" in engines and not ep.synth:
        summary["engine_E"] = _engine_arms(ep, entry, clean_tables, oracle_vals, rows, series, e_paths, e_arms)
    summary["series"] = series
    return {"rows": rows, "fields": fields, "summary": summary}


def _engine_arms(ep: Episode, entry: dict, clean_tables: dict, oracle_vals: dict, rows: list, series: dict,
                 e_paths: int, e_arms=None) -> dict:
    """The monthly engine's arms. clean_tables: {arm: the clean table the Q arm ran} (an arm Q could not run is not run
    on E either: the comparison needs both)."""
    from ...qmodel import qtable as QT
    from . import engines as EN
    out = {"arms": {}, "notes": []}
    try:
        st = EN.setup(ep.train_end)
    except EN.EngineSkip as err:
        out.update(skipped=True, reason=str(err))
        return out
    except Exception as err:  # noqa: BLE001
        out.update(skipped=True, reason=f"月度引擎拟合不出来：{type(err).__name__}: {err}")
        return out
    out["notes"] += st["notes"]
    out["model"] = st["spec_name"]
    stats = QT.stats_table(ep.Q, ep.train_end, shocks=True)
    tables = {"U": QT.minimal(ep.H)}
    t_notes: dict[str, list] = {}
    shas: dict[str, str] = {}
    for arm, vals in oracle_vals.items():
        tables[arm] = oracle_qtable(vals, ep.H)
    for arm, clean in clean_tables.items():
        shas[arm] = QT.digest(clean)
        tables[arm], t_notes[arm] = e_table(clean)
    for arm in [a for a in (e_arms or QE_ARMS) if a in tables]:
        try:
            spec, rep = EN.phase_spec(tables[arm], ep.Q, ep.train_end, ep.H, stats=stats)
            paths, info = EN.run(st, spec, ep.H, e_paths, ep.seed)
        except Exception as err:  # noqa: BLE001
            out["arms"][arm] = {"skipped": True, "reason": f"{type(err).__name__}: {err}"}
            continue
        for r in ST.score_draws(paths, ep.realized, ep.H):
            rows.append({"code": ep.code, "set": entry["set"], "bust": entry.get("bust"), "engine": "E", "arm": arm,
                         "variant": variant("E", arm), "cma": "historical", **r})
        series[variant("E", arm)] = _small_multiple(paths, ep.realized, ep.H)
        out["arms"][arm] = {"dropped": [d["field"] for d in rep["dropped"]][:20], "phases": len(info["phases"]),
                            "shrink": info["shrink"], "warnings": info["warnings"][:5],
                            **({"table_sha256": shas[arm]} if arm in shas else {}),
                            **({"notes": t_notes[arm]} if t_notes.get(arm) else {})}
    return out


# ================================================================ the run

def _translations_by_code(rid: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cid, rec in RN.translations(rid).items():
        out.setdefault(rec["code"], {})[rec["arm"]] = rec
    return out


def score_run(rid: str, *, n_paths: int | None = None, n_boot: int | None = None, engines=None, cmas=C.CMA_ARMS,
              codes=None, progress=None, cancel=None, Q: pd.DataFrame | None = None,
              e_paths: int = C.E_PATHS, e_arms=None) -> dict:
    """Simulate and score every episode of a run (after its translations); writes the run's score files and returns
    the stats (see the module docstring)."""
    from ...qmodel import panel as qpanel
    run = RN.read_run(rid)
    s = run["settings"]
    engines = list(engines or s.get("engines") or ["Q"])
    n_paths = int(n_paths or s.get("n_paths") or C.Q_PATHS)
    n_boot = s.get("n_boot") if n_boot is None else n_boot
    Q = qpanel.load_panel() if Q is None else Q
    key = K.load_key()
    trans = _translations_by_code(rid)
    order = sorted(set(s["episodes"]) | set(trans))
    if codes is not None:
        order = [c for c in order if c in set(codes)]
    reg_index = {c: i for i, c in enumerate(sorted(set(key["episodes"]) | set(key["synth"])))}
    RN.write_run(rid, status="scoring", progress={"phase": "score", "done": 0, "total": len(order)})
    rows, fields, summaries = [], [], {}
    t0 = time.perf_counter()
    for i, code in enumerate(order):
        if cancel is not None and cancel.is_set():
            RN.write_run(rid, status="cancelled")
            return {"status": "cancelled"}
        entry = K.episode(code)
        if "kind" not in entry:
            entry = {**entry, "kind": "episode"}
        res = score_episode(code, entry, trans.get(code, {}), Q=Q, seed=int(s.get("seed", C.SEED)) + reg_index[code],
                            n_paths=n_paths, n_boot=n_boot, engines=engines, e_paths=e_paths, cmas=cmas,
                            e_arms=e_arms)
        rows += res["rows"]
        fields += res["fields"]
        summaries[code] = res["summary"]
        RN.write_run(rid, progress={"phase": "score", "done": i + 1, "total": len(order)})
        if progress is not None:
            progress(i + 1, len(order))
    d = RN.run_dir(rid)
    R = pd.DataFrame(rows)
    Fd = pd.DataFrame(fields)
    R.to_csv(d / "score_rows.csv", index=False, float_format="%.8g")
    Fd.to_csv(d / "fields.csv", index=False, float_format="%.8g")
    windows = K.unsealed(order)
    for c, w in windows.items():
        summaries[c]["window"] = w
    out = summarize(R, windows, engines=engines)
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    out["settings"] = {"n_paths": n_paths, "n_boot": n_boot, "engines": engines, "cmas": list(cmas),
                       "e_paths": int(e_paths)}
    _dump(d / "episodes.json", summaries)
    _dump(d / "stats.json", out)
    return out


def restat(rid: str) -> dict:
    """stats.json again from the run's stored score_rows.csv (no simulation): after a change to the statistics or the
    T-N record's wording."""
    d = RN.run_dir(rid)
    R = pd.read_csv(d / "score_rows.csv")
    old = json.loads((d / "stats.json").read_text(encoding="utf-8")) if (d / "stats.json").exists() else {}
    windows = K.unsealed(sorted(R["code"].unique()))
    out = summarize(R, windows, engines=sorted(R["engine"].unique()))
    for k in ("elapsed_s", "settings"):
        if k in old:
            out[k] = old[k]
    _dump(d / "stats.json", out)
    return out


def _dump(path, obj) -> None:
    from ... import jsonio
    path.write_text(jsonio.dumps(obj), encoding="utf-8")


# ================================================================ statistics over episodes

def _sets(windows: dict) -> dict[str, list]:
    clean = sorted(c for c, w in windows.items() if w["set"] == "clean")
    return {"clean": clean, "short": sorted(c for c, w in windows.items() if w["set"] == "short"),
            "synth": sorted(c for c, w in windows.items() if w["set"] == "synth"),
            "bust": [c for c in clean if windows[c].get("bust")],
            "ordinary": [c for c in clean if not windows[c].get("bust")]}


def summarize(R: pd.DataFrame, windows: dict, *, engines=("Q",), head_cma: str = "historical") -> dict:
    """Pooled tables, paired comparisons, claims and the rest of stats.json from the score rows."""
    tn, tn_sha = load_tn()
    sets = _sets(windows)
    out = {"tn": {"id": tn["id"], "version": tn["version"], "sha256": tn_sha, "file": TN_FILE}, "sets": sets,
           "head_cma": head_cma}
    if R.empty:
        out.update(pooled={}, paired=[], claims=[], n_per_horizon={})
        return out
    out["pooled"] = {sname: ST.pooled(R, codes) for sname, codes in sets.items() if codes}
    out["n_per_horizon"] = {sname: {h: int(R[(R["code"].isin(codes)) & (R["horizon"] == h)]["code"].nunique())
                                    for h in C.HORIZONS} for sname, codes in sets.items() if codes}
    have = set(zip(R["variant"], R["cma"]))
    pairs = []

    def add(a, b, h, assets, codes, set_name, cma_a=head_cma, cma_b=head_cma):
        if (a, cma_a) not in have or (b, cma_b) not in have or not codes:
            return None
        p = ST.paired(R, (a, cma_a), (b, cma_b), h, assets, codes=codes)
        if p:
            p["set"] = set_name
            pairs.append(p)
        return p

    for set_name in ("clean", "short", "bust", "ordinary", "synth"):
        codes = sets.get(set_name) or []
        for eng in engines:
            for h in C.HORIZONS:
                for arm in PAIRS_VS_U + ("NS",):
                    add(variant(eng, arm), variant(eng, "U"), h, "equity9", codes, set_name)
                for a, b in EXTRA_PAIRS:
                    add(variant(eng, a), variant(eng, b), h, "equity9", codes, set_name)
            for h in ("4Q", "end"):
                for assets in ("us_equity", "all13", "gold", "crude", "rates"):
                    for arm in ("N1", "N2", "N3", "OB", "OF"):
                        add(variant(eng, arm), variant(eng, "U"), h, assets, codes, set_name)
                if set_name == "clean":
                    add(variant(eng, "V"), variant(eng, "U"), h, "us_equity", codes, set_name)
        if "E" in engines and set_name in ("clean", "short"):
            for h in C.HORIZONS:
                for arm in QE_ARMS:
                    add(variant("Q", arm), variant("E", arm), h, "equity9", codes, set_name)
        if set_name == "clean":
            for h in ("4Q", "end"):
                for arm in ("N2", "OB", "U"):
                    add(variant("Q", arm), variant("Q", arm), h, "equity9", codes, set_name, cma_a="no_rerate",
                        cma_b=head_cma)
    out["paired"] = pairs
    clean = sets["clean"]
    narr_tests = [p for p in pairs if p["set"] == "clean" and p["assets"] == "equity9" and p["b"].endswith(":U")
                  and p["a"].split(":")[1] in ("N1", "N2", "N3") and p["a_cma"] == head_cma]
    out["chance"] = ST.chance(narr_tests)
    out["claims"] = claims(pairs, tn, engines)
    out["decomposition"] = decomposition(R, clean, head_cma)
    out["discrimination"] = discrimination(R, clean, head_cma, pairs)
    out["ts_inputs"] = ts_inputs(R, pairs, clean, head_cma) if "E" in engines else None
    out["calibration"] = {eng: {arm: _pit_mad(R, variant(eng, arm), head_cma, clean)
                                for arm in QE_ARMS + ("NS",)} for eng in engines}
    return out


def _find(pairs, a, b, h, assets="equity9", set_name="clean"):
    for p in pairs:
        if p["a"] == a and p["b"] == b and p["horizon"] == h and p["assets"] == assets and p["set"] == set_name \
                and p["a_cma"] == p["b_cma"]:
            return p
    return None


def claims(pairs: list, tn: dict, engines) -> list[dict]:
    """The pre-registered claims C1–C4 (the T-N record): passed only when every condition holds."""
    out = []
    for c in tn["claims"]:
        if c.get("always"):
            out.append({"id": c["id"], "text": c["text"], "always": True})
            continue
        conds, ok, mdes = [], True, []
        for cond in c["all"]:
            p = _find(pairs, cond["a"], cond["b"], cond["horizon"], cond["assets"], cond["set"])
            if p is None or p.get("hi90") is None:
                conds.append({**cond, "evaluated": False})
                ok = False
                continue
            passed = p["hi90"] < cond["hi90_below"]
            ok = ok and passed
            if p.get("mde") is not None:
                mdes.append(p["mde"])
            conds.append({**cond, "evaluated": True, "diff": p["diff"], "lo90": p["lo90"], "hi90": p["hi90"],
                          "n": p["n_episodes"], "mde": p.get("mde"), "passed": passed})
        evaluated = all(x["evaluated"] for x in conds)
        mde = conds[0].get("mde") if conds and conds[0].get("evaluated") else None   # the primary comparison's
        text = c["text"] if ok and evaluated else (c["otherwise"].format(mde=f"{mde:.3f}" if mde is not None else "—")
                                                   if evaluated else f"未评估：{c['text']}（缺少需要的臂或片段）")
        out.append({"id": c["id"], "claim": c["text"], "passed": bool(ok and evaluated), "evaluated": evaluated,
                    "text": text, "mde": mde, "conditions": conds})
    return out


def decomposition(R: pd.DataFrame, codes, cma: str) -> dict:
    """(N − O-full) + (O-full − realized) per narrative arm, equity9, at 4Q and the window end; and O-B − O-A′."""
    out = {}
    sub = R[R["code"].isin(codes) & (R["cma"] == cma) & R["asset"].isin(C.EQUITY9) & (R["engine"] == "Q")]
    piv = sub.pivot_table(index=["code", "asset", "horizon"], columns="arm", values="q50", aggfunc="first")
    y = sub.groupby(["code", "asset", "horizon"])["y"].first()
    for h in ("4Q", "end"):
        for arm in ("N1", "N2", "N3", "NW"):
            if arm not in piv.columns or "OF" not in piv.columns:
                continue
            P = piv.xs(h, level="horizon")
            yy = y.xs(h, level="horizon")
            ok = P[arm].notna() & P["OF"].notna()
            if not ok.any():
                continue
            hyp = (P[arm] - P["OF"])[ok]
            sim = (P["OF"] - yy.reindex(P.index))[ok]
            tot = (P[arm] - yy.reindex(P.index))[ok]
            per = pd.DataFrame({"hyp": hyp, "sim": sim, "tot": tot}).groupby(level="code").mean()
            out[f"{arm}|{h}"] = {"n_episodes": int(len(per)), "total_mean": round(float(per["tot"].mean()), 6),
                                 "hypothesis_mean": round(float(per["hyp"].mean()), 6),
                                 "simulation_mean": round(float(per["sim"].mean()), 6),
                                 "total_mabs": round(float(per["tot"].abs().mean()), 6),
                                 "hypothesis_mabs": round(float(per["hyp"].abs().mean()), 6),
                                 "simulation_mabs": round(float(per["sim"].abs().mean()), 6),
                                 "per_episode": {c: {k: round(float(v), 6) for k, v in r.items()}
                                                 for c, r in per.iterrows()}}
    return out


def discrimination(R: pd.DataFrame, codes, cma: str, pairs: list) -> dict:
    """Right (N1) against wrong (N-wrong) narrative at the window end, equity9: per episode and paired."""
    sub = R[R["code"].isin(codes) & (R["cma"] == cma) & (R["horizon"] == "end") & R["asset"].isin(C.EQUITY9)
            & (R["engine"] == "Q") & R["arm"].isin(["N1", "NW"])]
    per = sub.groupby(["code", "arm"])["abs_err"].mean().unstack()
    rows = []
    if {"N1", "NW"} <= set(per.columns):
        for c, r in per.dropna().iterrows():
            rows.append({"code": c, "right_mae": round(float(r["N1"]), 6), "wrong_mae": round(float(r["NW"]), 6),
                         "right_better": bool(r["N1"] < r["NW"])})
    p = _find(pairs, "Q:N1", "Q:NW", "end") or {}
    return {"episodes": rows, "n": len(rows), "share_right_better": (round(float(np.mean([x["right_better"]
                                                                                           for x in rows])), 4)
                                                                    if rows else None),
            "right_minus_wrong": {k: p.get(k) for k in ("diff", "lo90", "hi90", "n_episodes", "mde")} if p else None}


def _pit_mad(R: pd.DataFrame, var: str, cma: str, codes) -> float | None:
    sub = R[(R["variant"] == var) & (R["cma"] == cma) & R["code"].isin(codes) & R["asset"].isin(C.EQUITY9)
            & (R["horizon"] == "end")]
    if sub.empty:
        return None
    v = sub.groupby("code")["pit"].apply(lambda s: (s - 0.5).abs().mean())
    v = v[np.isfinite(v)]
    return round(float(v.mean()), 6) if len(v) else None


def ts_inputs(R: pd.DataFrame, pairs: list, clean, cma: str) -> dict:
    """What the switch gate T-S (§4.6, P9) reads from this run: Q − E on the oracle arms and the narratives (equity9,
    4Q and the window end) and each engine's mean |PIT − 0.5|. The gate itself is P9's."""
    out = {"oracle": {}, "narrative": {}, "pit_mad": {}}
    for arm in ("OA", "OB", "OF"):
        for h in ("4Q", "end"):
            p = _find(pairs, variant("Q", arm), variant("E", arm), h)
            out["oracle"][f"{arm}|{h}"] = {k: p.get(k) for k in ("diff", "lo90", "hi90", "n_episodes")} if p else None
    for arm in ("N1", "N2", "N3"):
        for h in ("4Q", "end"):
            p = _find(pairs, variant("Q", arm), variant("E", arm), h)
            out["narrative"][f"{arm}|{h}"] = {k: p.get(k) for k in ("diff", "lo90", "hi90", "n_episodes")} \
                if p else None
    for eng in ("Q", "E"):
        vals = [x for x in (_pit_mad(R, variant(eng, a), cma, clean) for a in QE_ARMS) if x is not None]
        out["pit_mad"][eng] = round(float(np.mean(vals)), 6) if vals else None
    return out
