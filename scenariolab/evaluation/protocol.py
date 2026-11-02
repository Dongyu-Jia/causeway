"""The pre-registered runs of the quarterly stack: T-A, T-C and T-S (SIM_DESIGN §3.9 protocol.py; §4.2, §4.3, §4.6,
§5.8, §14 P5). The CLI is tools/run_preregistered.py; the opt-in build step `evaluation` is run() below.

    python tools/run_preregistered.py freeze [--date YYYY-MM-DD]   # write <date>_TA.json, _TC.json, _TS.json (frozen)
    python tools/run_preregistered.py freeze --only TC --tag r2 --supersedes 2026-11-02_TC.json   # one new record
    python tools/run_preregistered.py note TA --text "..."         # append a note (never edits the frozen part)
    python tools/run_preregistered.py run TA|TC [--date ...]       # run once; append the result to the record
    python tools/run_preregistered.py status TS [--date ...]       # append T-S's status (it needs P7's records)
    python tools/run_preregistered.py check [--inputs]             # verify every record (and the inputs' hashes)
    python tools/run_preregistered.py frozen-copy <record> <out>   # the record exactly as it was when frozen
    python tools/run_preregistered.py smoke                        # plumbing run on data <= 2014Q4 (prints no score)
    python -m scenariolab.build --only evaluation                  # integrity, inputs and a reproduction check

Rules (CONTRIBUTING: never edit a frozen pre-registration record):
  * A record is written once, by `freeze`, before anything it governs has run: protocol, candidates, thresholds, seeds
    and the sha256 of every input file (code, model card, data). `frozen_sha256` is the hash of its canonical JSON.
    The code is the runs' static import closure (qmodel.spec.code_files of this module and the CLI script), not the
    whole package: a file the runs never import cannot invalidate a record. The record also notes the git commit and
    whether the closure was committed there (commit the code first: reproduction then means checking out that commit).
  * `run` refuses a record whose frozen part does not match its hash, an input that differs from the frozen hash, or
    a record that already holds a run: each test runs exactly once. The result is APPENDED to `results` (with its own
    hash and the frozen hash it ran against); nothing earlier in the file changes, so `frozen-copy` rebuilds the
    frozen file byte for byte. A protocol change needs a new dated record that names the one it supersedes
    (freeze --only ... --tag ... --supersedes ...); a note (kind "note") is appended the same way and changes nothing
    else. Timestamps are compared as UTC instants.
  * Every number comes from the run; the derived rows (conditional paths, T-C rows) go to config_q.PREREG_RESULTS_DIR
    with their hashes in the result. No LLM is called and nothing is downloaded.

What each record governs:
  TA  the asset layer: the registered candidates C1-C3 against L0 (selection on 1990-2014, one test on 2015-26), the
      report-only rows (the masked NN, G_app = L0 + C1 + C2), and config_q.TA_DEFAULT, which the T-A rule sets
  TC  the factor layer's completion: the completion weight w and the regime covariance (qmodel.factor_checks.tc)
  TS  the switch gate: the monthly engine stays the default until it passes; it needs the narrative test (P7)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..qmodel import config_q as cq
from ..qmodel import spec as qspec

ROOT = config.ROOT
CLI = ROOT / "tools" / "run_preregistered.py"

SCHEMA_ID = "preregistered/1"
RECORD_IDS = ("TA", "TC", "TS")
TITLES = {"TA": "T-A：资产层候选项的预注册检验", "TC": "T-C：因子层补全的预注册检验",
          "TS": "T-S：季度模型替换月度引擎的切换关"}
RUNNABLE = {"TA": "run", "TC": "run", "TS": "status"}
KINDS = ("run", "status", "note")
JOINT = "joint"


# ================================================================ JSON and hashes

def jsonable(o):
    """numpy / pandas -> plain JSON values; NaN and inf -> None; tuples -> lists; keys -> str."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (pd.Timestamp, pd.Period)):
        return str(o)
    if isinstance(o, Path):
        return str(o)
    return o


def canonical(obj) -> bytes:
    return json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256_obj(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dumps_record(record: dict) -> str:
    return json.dumps(jsonable(record), ensure_ascii=False, indent=1, allow_nan=False) + "\n"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_record(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def instant(ts) -> datetime | None:
    """An ISO timestamp as an aware UTC datetime (None if unreadable): records written in different time zones compare
    by instant, never as strings."""
    try:
        t = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.astimezone()
    return t.astimezone(timezone.utc)


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ================================================================ inputs and environment

def code_files(root: Path = ROOT) -> list[Path]:
    """The code a run may execute: the static import closure of this module and the CLI script (every package module
    reachable from them, imports inside functions included; qmodel.spec.code_files)."""
    cli = root / "tools" / "run_preregistered.py"
    return qspec.code_files(entries=("scenariolab.evaluation.protocol",), root=root,
                            extra_paths=[cli] if cli.exists() else [])


def card_files() -> list[Path]:
    return [cq.CARD_PATH, cq.SCHEMA_PATH, cq.SCHEMA_VIEW_PATH]


def data_files() -> list[Path]:
    """The data the T-A and T-C runs read (data/ is git-ignored; the hashes identify the build)."""
    return [config.PANEL_PATH, config.PANEL_META_PATH, cq.PANEL_Q_PATH, cq.PANEL_Q_META_PATH,
            config.PROCESSED_DIR / "structural_shocks.csv", config.RESULTS_DIR / "model_selection.json",
            config.RAW_DIR / "shiller" / "ie_data.xls"]


def _digest(files: dict[str, str]) -> str:
    return hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode()).hexdigest()


def inputs_block() -> dict:
    """sha256 of every input file, by group, with one digest per group."""
    out = {}
    for group, files in (("code", code_files()), ("card", card_files()), ("data", data_files())):
        h = {rel(p): (sha256_file(p) if p.exists() else None) for p in files}
        out[group] = {"digest": _digest({k: v or "missing" for k, v in h.items()}), "files": h}
    meta = cq.PANEL_Q_META_PATH
    if meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8")).get("info", {})
        out["data"]["panel_quarterly_info"] = {k: info.get(k) for k in ("today", "last_data_quarter", "inputs_digest",
                                                                       "card_digest")}
    return out


def verify_inputs(frozen_inputs: dict) -> list[dict]:
    """Files whose current hash differs from the frozen one (an empty list: the inputs are those frozen). Code: the
    files of today's import closure of the runs -- each must equal its frozen hash, and one the record did not hash
    is new code; a frozen file the runs no longer import is not compared (the first records, 2026-11-02, hashed the
    whole package: server.py and the like changing must not void them)."""
    bad = []
    for group in ("code", "card", "data"):
        want = frozen_inputs[group]["files"]
        have = {rel(p): (sha256_file(p) if p.exists() else None)
                for p in (code_files() if group == "code" else card_files() if group == "card" else data_files())}
        keys = set(have) if group == "code" else set(want) | set(have)
        for f in sorted(keys):
            if f not in want:
                bad.append({"group": group, "file": f, "frozen": None, "now": have.get(f), "status": "new"})
            elif want.get(f) != have.get(f):
                bad.append({"group": group, "file": f, "frozen": want.get(f), "now": have.get(f),
                            "status": "changed" if have.get(f) else "missing"})
    return bad


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def code_commit() -> dict:
    """The commit the runs' code is at: HEAD, and whether every file of the import closure is committed unchanged there
    (a frozen record is reproducible by checking out that commit only if it is)."""
    files = [rel(p) for p in code_files()]
    status = _git("status", "--porcelain", "--", *files)
    dirty = sorted({ln.strip().split(maxsplit=1)[-1] for ln in (status or "").splitlines() if ln.strip()})
    return {"head": _git("rev-parse", "HEAD"), "closure_files": len(files), "closure_clean": status is not None and
            not dirty, "dirty_closure_files": dirty,
            "rule": "commit the code before freezing; reproduction = git checkout <head> (no squash merge of the "
                    "record commits)"}


def environment() -> dict:
    import scipy
    status = _git("status", "--porcelain") or ""
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(), "git_head": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_entries": len([s for s in status.splitlines() if s.strip()]),
            "code_commit": code_commit(), "llm": "none (no model is called)", "downloads": "none"}


# ================================================================ the record lifecycle

def record_path(date: str, rid: str, directory: Path | None = None, tag: str = "") -> Path:
    """<date>_<rid>.json, or <date>-<tag>_<rid>.json for a later record of the same date (a superseding one)."""
    return Path(directory or cq.PREREG_DIR) / f"{date}{'-' + tag if tag else ''}_{rid}.json"


def make_record(rid: str, date: str, protocol: dict, inputs: dict, env: dict, supersedes: dict | None = None) -> dict:
    at = now()
    body = {"record": rid, "date": date, "frozen_at": at, "protocol": protocol, "inputs": inputs, "environment": env,
            "rules": {"append_only": "results are appended; the frozen part never changes",
                      "runs": "exactly once" if RUNNABLE[rid] == "run" else
                      "status entries now; the evaluation is P9's run",
                      "inputs_must_match": rid in ("TA", "TC"),
                      "supersede": "a protocol change is a new dated record that names this one"}}
    if supersedes:
        body["supersedes"] = supersedes
    frozen = jsonable(body)
    return {"schema": SCHEMA_ID, "record": rid, "date": date, "title": TITLES[rid], "frozen_at": at,
            "frozen": frozen, "frozen_sha256": sha256_obj(frozen), "results": []}


def freeze(date: str, directory: Path | None = None, protocols: dict | None = None, only=None, tag: str = "",
           supersedes: str | None = None) -> dict[str, Path]:
    """Write the frozen records (all three, or `only` some, under <date>-<tag>); refuse if one exists already (a record
    is written once). supersedes: the file name of the record each new one replaces (it must exist and be intact; its
    frozen hash is copied into the new record's frozen part)."""
    ids = tuple(only or RECORD_IDS)
    paths = {rid: record_path(date, rid, directory, tag) for rid in ids}
    exist = [str(p) for p in paths.values() if p.exists()]
    if exist:
        raise SystemExit(f"拒绝覆盖已冻结的记录：{exist}")
    old = None
    if supersedes:
        op = Path(directory or cq.PREREG_DIR) / supersedes
        if not op.exists():
            raise SystemExit(f"被取代的记录不存在：{op}")
        orec = load_record(op)
        probs = check_record(orec)
        if probs:
            raise SystemExit(f"被取代的记录不完整：{probs}")
        if len(ids) != 1 or orec["record"] != ids[0]:
            raise SystemExit(f"--supersedes 只能用于同一种记录（{orec['record']}），一次一个")
        old = {"record": op.name, "frozen_sha256": orec["frozen_sha256"], "frozen_at": orec["frozen_at"],
               "results": [(e["n"], e["kind"], e["result_sha256"]) for e in orec["results"]]}
    default = {"TA": ta_protocol, "TC": tc_protocol, "TS": lambda: ts_protocol(date)}
    protocols = protocols or {rid: default[rid]() for rid in ids}
    inputs, env = inputs_block(), environment()
    for rid, p in paths.items():
        write_atomic(p, dumps_record(make_record(rid, date, protocols[rid], inputs, env, old)))
    return paths


def frozen_copy(record: dict) -> str:
    """The record's text as `freeze` wrote it: the same file with `results` empty."""
    return dumps_record({**record, "results": []})


def check_record(record: dict) -> list[str]:
    """Integrity problems of one record (an empty list: it is intact)."""
    probs = []
    for k in ("schema", "record", "date", "title", "frozen_at", "frozen", "frozen_sha256", "results"):
        if k not in record:
            probs.append(f"missing key {k}")
    if probs:
        return probs
    if record["schema"] != SCHEMA_ID:
        probs.append(f"schema {record['schema']!r}")
    if sha256_obj(record["frozen"]) != record["frozen_sha256"]:
        probs.append("the frozen part does not match frozen_sha256 (it was edited)")
    if (record["frozen"].get("record") != record["record"] or record["frozen"].get("date") != record["date"]
            or record["frozen"].get("frozen_at") != record["frozen_at"]):
        probs.append("the header does not match the frozen part")
    runs = 0
    prev = instant(record["frozen_at"])
    if prev is None:
        probs.append("frozen_at is not an ISO timestamp")
    for i, e in enumerate(record["results"], 1):
        if e.get("n") != i:
            probs.append(f"result {i}: n = {e.get('n')}")
        if e.get("kind") not in KINDS:
            probs.append(f"result {i}: unknown kind {e.get('kind')!r}")
        if e.get("frozen_sha256") != record["frozen_sha256"]:
            probs.append(f"result {i}: ran against another frozen part")
        if sha256_obj(e.get("result")) != e.get("result_sha256"):
            probs.append(f"result {i}: the result does not match result_sha256 (it was edited)")
        t = instant(e.get("started_at"))
        if t is None or prev is None or t < prev:
            probs.append(f"result {i}: started before the record was frozen or before the previous result")
        prev = instant(e.get("finished_at", e.get("started_at"))) or prev
        runs += e.get("kind") == "run"
    if RUNNABLE[record["record"]] == "run" and runs > 1:
        probs.append(f"{runs} runs: each pre-registered test runs once")
    return probs


def append_result(path: Path, kind: str, result: dict, started_at: str, inputs_check: dict) -> dict:
    """Append one result entry; refuse a second run, a broken record, or a kind the record does not take."""
    rec = load_record(path)
    probs = check_record(rec)
    if probs:
        raise SystemExit(f"记录 {path} 不完整，拒绝追加：{probs}")
    if kind not in KINDS:
        raise ValueError(kind)
    if kind == "run" and RUNNABLE[rec["record"]] != "run":
        raise SystemExit(f"{rec['record']} 现在只能追加状态（status），评估在 P9")
    if kind == "run" and any(e.get("kind") == "run" for e in rec["results"]):
        raise SystemExit(f"{rec['record']} 已经运行过一次：预注册检验只运行一次")
    res = jsonable(result)
    entry = {"n": len(rec["results"]) + 1, "kind": kind, "started_at": started_at, "finished_at": now(),
             "frozen_sha256": rec["frozen_sha256"], "inputs_check": jsonable(inputs_check), "result": res,
             "result_sha256": sha256_obj(res)}
    rec["results"].append(entry)
    write_atomic(path, dumps_record(rec))
    return entry


def check_all(directory: Path | None = None, inputs: bool = False) -> dict:
    out = {}
    for p in sorted(Path(directory or cq.PREREG_DIR).glob("*_T?.json")):
        rec = load_record(p)
        r = {"problems": check_record(rec), "results": [(e["n"], e["kind"]) for e in rec.get("results", [])]}
        if inputs and rec.get("record") in ("TA", "TC"):
            r["inputs_now_differ"] = len(verify_inputs(rec["frozen"]["inputs"]))
        out[p.name] = r
    return out


# ================================================================ T-A: the asset-layer models

def _assets():
    from scenariolab.qmodel import assets as qa
    return qa


def _harness():
    from scenariolab.evaluation import harness as eh
    return eh


def make_joint_layer():
    """JointCandidateLayer: the asset layer with the union of its candidates' terms in ONE ridge per asset. For one
    candidate it is qmodel.assets.AssetLayer exactly; for several, AssetLayer would add separately fitted ridges to
    the same L0 residual, double counting what their terms share."""
    qa = _assets()

    class JointCandidateLayer(qa.AssetLayer):
        def _terms(self, frame, asset, states=None):
            out = {}
            for cid in self.candidates:
                out.update(qa.candidate_terms(cid, frame, asset, states))
            return out

        def _fit_candidates(self, Qt, E):
            frame = Qt.copy()
            frame["d_ip_lead"] = Qt["d_ip"].shift(-1)
            frame["d_unrate_lead"] = Qt["d_unrate"].shift(-1)
            first = {}
            for cid in self.candidates:
                first.update(cq.TA_CANDIDATES[cid].get("first", {}))
            out = {}
            for a in qa.RISK:
                terms = self._terms(frame, a)
                for c, q0 in first.items():
                    if c in terms:
                        terms[c] = np.where(frame.index >= pd.Period(q0, "Q"), terms[c], 0.0)
                names = list(terms)
                M = np.column_stack([terms[c] for c in names])
                y = E[a].reindex(frame.index).to_numpy(float)
                ok = np.isfinite(y) & np.isfinite(M).all(axis=1)
                mu, sd = M[ok].mean(0), M[ok].std(0)
                sd = np.where(sd > 1e-12, sd, 1.0)
                Z = np.clip((M[ok] - mu) / sd, -cq.TA_CLIP_SD, cq.TA_CLIP_SD)
                b0, b, lam = qa._ridge_cv(Z, y[ok])
                out[(JOINT, a)] = qa.CandidateFit("+".join(self.candidates), a, names, mu, sd, b0, b, lam)
            return out

        def _cand_mean(self, frame, states, shape):
            out = np.zeros(tuple(shape) + (len(qa.RISK),))
            if not self.candidates:
                return out
            for j, a in enumerate(qa.RISK):
                cf = self.cand_fits[(JOINT, a)]
                out[..., j] += np.broadcast_to(cf.predict(self._terms(frame, a, states)), shape)
            return out

    return JointCandidateLayer


def make_window_lead_fitted():
    """QuarterlyFitted for C3 under the harness rule: the realized next-quarter leads inside the window; the window's
    last quarter's lead lies outside it and is set to the lead's training mean (the model's unconditional value)."""
    eh = _harness()

    class WindowLeadFitted(eh.QuarterlyFitted):
        def __init__(self, layer, Q, lead_fill: dict, **kw):
            super().__init__(layer, Q, **kw)
            self.lead_fill = dict(lead_fill)

        def _inputs(self, origin, months):
            q0, H, X = super()._inputs(origin, months)
            X = X.copy()
            for c, v in self.lead_fill.items():
                if c in X.columns:
                    X.iloc[-1, X.columns.get_loc(c)] = v
            return q0, H, X

    return WindowLeadFitted


def lead_fill_of(layer) -> dict:
    """The training means of the C3 lead terms in the us_equity equation's sample."""
    cf = layer.cand_fits[(JOINT, "us_equity")]
    return {c: float(m) for c, m in zip(cf.names, cf.mu) if c in ("d_ip_lead", "d_unrate_lead")}


def joint_model_fn(Q: pd.DataFrame, candidates=(), *, name: str, lead_rule: str = "window", cma: str = "historical",
                   n_boot: int = cq.PARAM_BOOT, seed: int = 0, fit_log: list | None = None):
    """model_fn for the harness: L0 (+ candidates in one ridge) fitted at each train end (a quarter-end month) on the
    monthly train panel and the quarterly panel up to that quarter. lead_rule (C3): "window" (the harness rule) or
    "oracle" (the realized lead of the quarter after the window: outside the rule, report-only)."""
    eh = _harness()
    Layer = make_joint_layer()
    WLF = make_window_lead_fitted()
    cands = tuple(candidates)

    def fn(train, train_end):
        te = pd.Timestamp(train_end)
        if te.month % 3:
            raise ValueError("the quarterly asset layer needs a quarter-end train end")
        layer = Layer.fit(pd.Period(te, "Q"), panel=train, Q=Q, n_boot=n_boot, candidates=cands, seed=seed)
        if "C3" in cands and lead_rule == "window":
            fitted = WLF(layer, Q, lead_fill_of(layer), cma=cma, name=name)
        else:
            fitted = eh.QuarterlyFitted(layer, Q, cma=cma, name=name)
        if fit_log is not None:
            rec = {"model": name, "train_end": str(pd.Period(te, "Q"))}
            if cands:
                rec["ridge_lambda"] = {a: float(cf.lam) for (_, a), cf in layer.cand_fits.items()}
                rec["n_terms"] = {a: len(cf.names) for (_, a), cf in layer.cand_fits.items()}
            fit_log.append(rec)
        return fitted

    fn.model_name = name
    return fn


# ================================================================ T-A: the masked neural network (report-only row)
# A port of the research qmodel's qcore.MaskedNN (sim_redesign/qmodel/qcore.py) onto the app's quarterly panel, with
# the research harness adapter's path recursion (qeval.path_block) for windows longer than a quarter.

NN_STATES = ["s_ff", "s_term", "s_spread", "s_real_y10", "s_core_yoy", "s_infl_trend", "s_unrate", "s_sahm", "s_pd",
             "s_cape", "s_ecyc", "s_fcfy", "s_oil_real", "s_usd_real", "s_vol", "s_zlb", "s_core4"]
NN_ROLL = ["s_pd", "s_cape", "s_fcfy", "s_vol"]        # asset-derived states rolled with each simulated path
NN_HELD = ["s_ecyc"]                                   # held at the origin (Δlog Ê is not an output of the app's NN)
NN_CFG = {"hidden": 24, "alphas": (0.03, 0.1, 0.3, 1.0, 3.0), "K": 8, "w_x": 0.5, "maxiter": 300, "n_seeds": 4,
          "val_share": 0.8, "clip": 4.0, "inflate_bounds": (1.0, 2.0), "train_start": "1973Q2", "seed": 0,
          "min_obs": 12, "n_sim": 300, "pool_seed": 99, "val_mask_seed": 321}
RVARS = [f"y_{a}" for a in cq.ASSETS] + ["y_mult"]
E1_GROUP = ("E1", "E1c", "E2")
E1_ONLY = ("E1", "E2")


def nn_xvars() -> tuple[list[str], dict]:
    """The NN's x inputs (the 52 card fields with x_oil_cause carried by d_oil_sup, + the 5 policy shocks) and their
    endogeneity classes (d_oil_sup and the shocks count as E0, as in the research)."""
    from scenariolab.qmodel import card as qcard
    card = qcard.load_card()
    rows = {r["id"]: r for r in card["rows"]}
    xs = [f for f in qcard.x_fields(card) if f != "x_oil_cause"] + ["d_oil_sup"] + list(cq.SHOCKS)
    endo = {f: (rows[f].get("endo_code") if f in rows else "E0") for f in xs}
    endo.update({"d_oil_sup": "E0", **{u: "E0" for u in cq.SHOCKS}})
    return xs, endo


def with_shocks(Q: pd.DataFrame, train_end) -> tuple[pd.DataFrame, dict]:
    """The quarterly panel with the policy shocks u_* of the reaction functions fitted to train_end (qmodel.policy),
    and the fits."""
    from scenariolab.qmodel import policy
    fits = policy.fit(Q, str(pd.Period(train_end, "Q")))
    sh = policy.shocks(Q, fits)
    D = Q.copy()
    for u in cq.SHOCKS:
        D[u] = sh[u].reindex(D.index)
    return D, fits


def eq4_coefficients(fits: dict) -> dict[str, float]:
    """{shock: coefficient on eq4_lag} of the systematic equations that read the previous four quarters' equity
    returns (d_deficit -> u_def, d_perstax -> u_ptax)."""
    out = {}
    for f, ft in fits.items():
        if ft.identified and "eq4_lag" in ft.regressors:
            out[ft.shock] = float(ft.coef["eq4_lag"])
    return out


class _Std:
    def __init__(self, df: pd.DataFrame, clip: float):
        self.mu = df.mean()
        self.sd = df.std().replace(0, np.nan).fillna(1.0)
        self.cols = list(df.columns)
        self.clip = clip

    def z(self, df: pd.DataFrame) -> np.ndarray:
        return ((df[self.cols] - self.mu) / self.sd).to_numpy(float)

    def zc(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.z(df), -self.clip, self.clip)


class NNPool:
    """Training residual vectors (cross-asset correlation kept), divided by h(s_vol) = exp(b log(s_vol / median)),
    b fitted on the equity residuals and clipped to [0, 1]; draws are rescaled by h and by `scale` (the inflation)."""

    def __init__(self, E: np.ndarray, svol: np.ndarray, eq_cols: list[int]):
        ok = np.isfinite(E).all(axis=1) & np.isfinite(svol) & (svol > 0)
        E, sv = E[ok], svol[ok]
        self.med = float(np.median(sv))
        y = np.log(np.abs(E[:, eq_cols]) + 1e-6).mean(axis=1)
        x = np.log(sv / self.med)
        b = float(np.polyfit(x, y, 1)[0]) if len(x) > 10 else 0.0
        self.b = float(np.clip(b, 0.0, 1.0))
        self.Z = E / self._h(sv)[:, None]
        self.Z = self.Z - self.Z.mean(0)
        self.scale = np.ones(E.shape[1])

    def _h(self, sv):
        sv = np.where(np.isfinite(sv) & (sv > 0), sv, self.med)
        return np.exp(self.b * np.log(sv / self.med))

    def draw(self, n: int, svol: np.ndarray, rng) -> np.ndarray:
        """(n, T, R) draws for T quarters (or T paths) with state volatility svol (T,)."""
        svol = np.asarray(svol, float)
        idx = rng.integers(0, len(self.Z), size=(n, len(svol)))
        return self.Z[idx] * self._h(svol)[None, :, None] * self.scale[None, None, :]


class MaskedNN:
    """Masked MLP: [s, x * m, m] -> [r, x] (standardized); one tanh layer plus a linear skip from [s, x * m]; loss =
    weighted MSE (returns always, x only where masked but measured, weight w_x) + alpha/2 ||weights||^2; training set
    = K copies of every training quarter (copy 0 unmasked, the rest with random masks); full-batch L-BFGS-B; alpha by
    time-ordered validation (first val_share of the training quarters, scored on full, half-masked and O-A inputs);
    an ensemble over n_seeds seeds. Fitted on quarters <= train_end only."""

    def __init__(self, **cfg):
        self.cfg = {**NN_CFG, **cfg}

    # -- network
    def _shapes(self):
        return [("W1", (self.din, self.cfg["hidden"])), ("b1", (self.cfg["hidden"],)),
                ("W2", (self.cfg["hidden"], self.dout)), ("b2", (self.dout,)), ("Ws", (self.dsk, self.dout))]

    def _unpack(self, th):
        out, i = {}, 0
        for k, shp in self._shapes():
            n = int(np.prod(shp))
            out[k] = th[i:i + n].reshape(shp)
            i += n
        return out

    def _init(self, rng):
        parts = []
        for k, shp in self._shapes():
            if k == "W1":
                parts.append(rng.normal(scale=0.5 / math.sqrt(shp[0]), size=shp).ravel())
            elif k == "W2":
                parts.append(rng.normal(scale=0.01, size=shp).ravel())
            else:
                parts.append(np.zeros(int(np.prod(shp))))
        return np.concatenate(parts)

    def _forward(self, w, Xin):
        H = np.tanh(Xin @ w["W1"] + w["b1"])
        return H @ w["W2"] + w["b2"] + Xin[:, :self.dsk] @ w["Ws"], H

    def _loss_grad(self, th, Xin, Y, W, alpha):
        """Penalized weighted MSE and its gradient (the research's analytic backpropagation)."""
        sw = max(float(W.sum()), 1.0)
        w = self._unpack(th)
        out, H = self._forward(w, Xin)
        E = out - Y
        loss = float(np.sum(W * E * E) / sw)
        G = 2.0 * W * E / sw
        g = {"W2": H.T @ G, "b2": G.sum(0), "Ws": Xin[:, :self.dsk].T @ G}
        GH = (G @ w["W2"].T) * (1.0 - H * H)
        g["W1"] = Xin.T @ GH
        g["b1"] = GH.sum(0)
        for k in ("W1", "W2", "Ws"):
            loss += 0.5 * alpha * float(np.sum(w[k] ** 2))
            g[k] = g[k] + alpha * w[k]
        return loss, np.concatenate([g[k].ravel() for k, _ in self._shapes()])

    def _train(self, Xin, Y, W, alpha, rng):
        from scipy.optimize import minimize
        res = minimize(self._loss_grad, self._init(rng), args=(Xin, Y, W, alpha), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.cfg["maxiter"]})
        return res.x

    # -- data
    def _inputs(self, Sz, Xz, Mz):
        return np.concatenate([Sz, np.where(Mz, Xz, 0.0), Mz.astype(float)], axis=1)

    def _masks(self, avail, rng):
        n, K = avail.shape
        M = np.ones((n, K), bool)
        kind = rng.random(n)
        rate = rng.random(n)
        iid = rng.random((n, K)) < rate[:, None]
        M[kind < 0.7] = ~iid[kind < 0.7]
        sel = (kind >= 0.7) & (kind < 0.85)
        M[np.ix_(sel, self.e1)] = False
        sel2 = kind >= 0.85
        M[np.ix_(sel2, self.e1o)] = False
        return M & avail

    def _augment(self, Sz, Xz, avail, Rz, ravail, rng):
        Xs, Ys, Ws = [], [], []
        for k in range(self.cfg["K"]):
            M = avail.copy() if k == 0 else self._masks(avail, rng)
            Xs.append(self._inputs(Sz, Xz, M))
            Ys.append(np.concatenate([Rz, Xz], axis=1))
            Ws.append(np.concatenate([ravail, self.cfg["w_x"] * (avail & ~M)], axis=1).astype(float))
        return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ws)

    def _val_loss(self, th, Sz, Xz, avail, Rz, ravail):
        w = self._unpack(th)
        rng = np.random.default_rng(self.cfg["val_mask_seed"])
        tot = 0.0
        R = len(RVARS)
        for M in (avail, avail & (rng.random(avail.shape) >= 0.5), avail & ~self.e1[None, :]):
            out, _ = self._forward(w, self._inputs(Sz, Xz, M))
            tot += float(np.sum(ravail * (out[:, :R] - Rz) ** 2) / ravail.sum())
        return tot / 3.0

    def fit(self, Q: pd.DataFrame, train_end) -> "MaskedNN":
        t0 = time.time()
        c = self.cfg
        te = pd.Period(train_end, "Q")
        D, fits = with_shocks(Q, te)
        self.eq4_coef = eq4_coefficients(fits)
        self.eq4_real = D["y_us_equity"].rolling(4).sum().shift(1)
        T = D.loc[pd.Period(c["train_start"], "Q"):te]
        T = T[T["y_us_equity"].notna()]
        xs, endo = nn_xvars()
        self.svars = list(NN_STATES) + [f"m1_{a}" for a in cq.ASSETS] + [f"m4_{a}" for a in cq.ASSETS]
        self.xvars = [v for v in xs if v in T.columns and T[v].notna().sum() >= c["min_obs"] and T[v].std() > 0]
        self.e1 = np.array([endo[v] in E1_GROUP for v in self.xvars])
        self.e1o = np.array([endo[v] in E1_ONLY for v in self.xvars])
        self.sst, self.xst = _Std(T[self.svars], c["clip"]), _Std(T[self.xvars], c["clip"])
        self.rst = _Std(T[RVARS], c["clip"])
        Sz = np.nan_to_num(self.sst.zc(T[self.svars]))
        Xz = self.xst.zc(T[self.xvars])
        avail = np.isfinite(Xz)
        Xz = np.nan_to_num(Xz)
        Rz = self.rst.z(T[RVARS])
        ravail = np.isfinite(Rz)
        Rz = np.nan_to_num(Rz)
        self.din = Sz.shape[1] + 2 * Xz.shape[1]
        self.dsk = Sz.shape[1] + Xz.shape[1]
        self.dout = len(RVARS) + Xz.shape[1]
        n = len(T)
        cut = int(c["val_share"] * n)
        seed = int(c["seed"])
        rng = np.random.default_rng(1000 * seed + 77)
        Xa, Ya, Wa = self._augment(Sz[:cut], Xz[:cut], avail[:cut], Rz[:cut], ravail[:cut], rng)
        self.val, ths = {}, {}
        for al in c["alphas"]:
            th = self._train(Xa, Ya, Wa, al, np.random.default_rng(1000 * seed + 78))
            ths[al] = th
            self.val[al] = self._val_loss(th, Sz[cut:], Xz[cut:], avail[cut:], Rz[cut:], ravail[cut:])
        self.alpha = min(self.val, key=self.val.get)
        R = len(RVARS)
        w = self._unpack(ths[self.alpha])
        o_tr = self._forward(w, self._inputs(Sz[:cut], Xz[:cut], avail[:cut]))[0][:, :R]
        o_va = self._forward(w, self._inputs(Sz[cut:], Xz[cut:], avail[cut:]))[0][:, :R]
        r_tr = np.sqrt(np.sum(ravail[:cut] * (o_tr - Rz[:cut]) ** 2, 0) / np.maximum(ravail[:cut].sum(0), 1))
        r_va = np.sqrt(np.sum(ravail[cut:] * (o_va - Rz[cut:]) ** 2, 0) / np.maximum(ravail[cut:].sum(0), 1))
        self.inflate = np.clip(r_va / np.maximum(r_tr, 1e-6), *c["inflate_bounds"])
        self.thetas = []
        for sd_ in range(int(c["n_seeds"])):
            rng = np.random.default_rng(1000 * seed + sd_)
            Xa, Ya, Wa = self._augment(Sz, Xz, avail, Rz, ravail, rng)
            self.thetas.append(self._train(Xa, Ya, Wa, self.alpha, rng))
        out = np.mean([self._forward(self._unpack(th), self._inputs(Sz, Xz, avail))[0][:, :R] for th in self.thetas],
                      axis=0)
        E = np.where(ravail, (Rz - out) * self.rst.sd.to_numpy(float), 0.0)
        eq = [RVARS.index(f"y_{a}") for a in cq.EQUITY_LIKE]
        self.pool = NNPool(E, T["s_vol"].to_numpy(float), eq)
        self.pool.scale = self.inflate
        self.g_d = float(T["y_dg"].mean())                                     # Δlog D12 (CRSP)
        self.g_e10 = float((T["y_fund"] - T["d_cpi"] / 400.0).mean())          # Δlog E10 real
        self.D, self.train_end, self.n_train, self.cut = D, te, n, cut
        self.seconds = round(time.time() - t0, 2)
        return self

    def predict_mean(self, S: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        """(rows, 14) the ensemble's mean returns for full (measured) x."""
        Sz = np.nan_to_num(self.sst.zc(S[self.svars]))
        Xz = self.xst.zc(X[self.xvars])
        Mz = np.isfinite(Xz)
        Xin = self._inputs(Sz, np.nan_to_num(Xz), Mz)
        outs = np.array([self._forward(self._unpack(th), Xin)[0][:, :len(RVARS)] for th in self.thetas])
        per = outs * self.rst.sd.to_numpy(float) + self.rst.mu.to_numpy(float)
        return per.mean(0)

    def info(self) -> dict:
        return {"train_end": str(self.train_end), "n_train": int(self.n_train),
                "n_validation": int(self.n_train - self.cut),
                "alpha": float(self.alpha), "val": {str(k): float(v) for k, v in self.val.items()},
                "inflate": {r: float(v) for r, v in zip(RVARS, self.inflate)}, "n_x": len(self.xvars),
                "n_s": len(self.svars), "pool_b": self.pool.b, "seconds": self.seconds}


def make_nn_fitted():
    eh = _harness()

    class NNFitted(eh.Fitted):
        """The masked NN as a harness model (oracle macro): 1Q = the ensemble mean on the origin's start states and
        the quarter's realized x (+ residual draws for the band); longer windows recursive as the research harness
        adapter: the realized x and macro states of every window quarter, the asset-derived states rolled with each
        simulated path, the window's prediction = the mean of n_sim simulated paths. After the origin, u_def and
        u_ptax (whose systematic parts read the previous four quarters' equity returns) are recomputed from the path's
        simulated returns, so no realized return of the window reaches an input (the research adapter used the
        realized ones)."""

        def __init__(self, nn: MaskedNN, name: str = "NN"):
            self.nn, self.name = nn, name
            self.D = nn.D
            self.assets = list(cq.ASSETS)
            self.train_end = pd.Period(nn.train_end, "Q").end_time.to_period("M").to_timestamp()
            self.cols = [RVARS.index(f"y_{a}") for a in self.assets]

        def _frame(self, q, cols, n):
            return pd.DataFrame(np.repeat(self.D.loc[[q], cols].to_numpy(float), n, axis=0), columns=cols)

        def _sim(self, q0: pd.Period, H: int, n: int, seed: int) -> np.ndarray:
            D, nn = self.D, self.nn
            qs = pd.period_range(q0, periods=H, freq="Q")
            out = np.full((n, H, len(RVARS)), np.nan)
            if any(q not in D.index for q in qs):
                return out
            rng = np.random.default_rng(seed)
            eq = RVARS.index("y_us_equity")
            st = {c: np.full(n, float(D.at[q0, c])) for c in NN_ROLL}
            for a in cq.ASSETS:
                st[f"m1_{a}"] = np.full(n, float(D.at[q0, f"m1_{a}"]))
                st[f"m4_{a}"] = np.full(n, float(D.at[q0, f"m4_{a}"]))
            hist = {a: np.tile(np.array([float(D.at[q0 - k, f"y_{a}"]) if (q0 - k) in D.index else np.nan
                                         for k in (3, 2, 1)]), (n, 1)) for a in cq.ASSETS}
            prev_eq = np.full(n, float(D.at[q0 - 1, "y_us_equity"]) if (q0 - 1) in D.index else 0.0)
            held = {c: float(D.at[q0, c]) for c in NN_HELD}
            for k, q in enumerate(qs):
                S = self._frame(q, nn.svars, n)
                if k > 0:
                    for c, v in st.items():
                        S[c] = v
                    for c, v in held.items():
                        S[c] = v
                X = self._frame(q, nn.xvars, n)
                if k > 0:
                    e4 = float(nn.eq4_real.get(q, np.nan))
                    for u, b in nn.eq4_coef.items():
                        if u in X.columns:
                            X[u] = X[u].to_numpy(float) + b * (e4 - st["m4_us_equity"])
                r = nn.predict_mean(S, X) + nn.pool.draw(1, S["s_vol"].to_numpy(float), rng)[0]
                out[:, k, :] = r
                pi = float(D.at[q, "d_cpi"]) / 400.0
                g = float(D.at[q, "d_rgdp"]) / 400.0 + pi if np.isfinite(D.at[q, "d_rgdp"]) else pi + 0.006
                y_eq = r[:, eq]
                st["s_pd"] = st["s_pd"] + y_eq - nn.g_d
                st["s_cape"] = st["s_cape"] + y_eq - pi - nn.g_e10
                st["s_fcfy"] = st["s_fcfy"] * np.exp(g - y_eq)
                st["s_vol"] = np.sqrt((y_eq ** 2 + prev_eq ** 2) / 2.0)
                prev_eq = y_eq
                for a in cq.ASSETS:
                    j = RVARS.index(f"y_{a}")
                    h = np.concatenate([hist[a], r[:, [j]]], axis=1)
                    st[f"m1_{a}"] = h[:, -1]
                    st[f"m4_{a}"] = h[:, -4:].sum(axis=1)
                    hist[a] = h[:, -3:]
            return out

        def predict_path(self, vis, origin, months):
            q0, H = eh._quarter_window(origin, months)
            o = pd.Timestamp(origin)
            if H == 1:
                if q0 not in self.D.index:
                    mean = np.full((1, len(RVARS)), np.nan)
                else:
                    mean = self.nn.predict_mean(self._frame(q0, self.nn.svars, 1), self._frame(q0, self.nn.xvars, 1))
            else:
                mean = self._sim(q0, H, int(self.nn.cfg["n_sim"]), 7 + o.year * 12 + o.month).mean(axis=0)
            idx = pd.date_range(o, periods=int(months), freq="MS")
            return pd.DataFrame(eh._in_last_months(mean[:, self.cols], origin, months, self.assets), index=idx,
                                columns=self.assets)

        def simulate_paths(self, vis, origin, months, n_paths, seed):
            q0, H = eh._quarter_window(origin, months)
            sims = self._sim(q0, H, int(n_paths), int(seed))
            return eh._in_last_months(sims[:, :, self.cols], origin, months, self.assets)

    return NNFitted


def nn_model_fn(Q: pd.DataFrame, *, name: str = "NN", cfg: dict | None = None, fit_log: list | None = None):
    NNFitted = make_nn_fitted()

    def fn(train, train_end):
        te = pd.Period(pd.Timestamp(train_end), "Q")
        nn = MaskedNN(**(cfg or {})).fit(Q, te)
        if fit_log is not None:
            fit_log.append({"model": name, **nn.info()})
        return NNFitted(nn, name=name)

    fn.model_name = name
    return fn


# ================================================================ T-A: protocol

TA_MODELS = {
    # name: (candidates, role, lead rule)
    "L0q": ((), "baseline", None),
    "L0q+C1": (("C1",), "candidate", None),
    "L0q+C2": (("C2",), "candidate", None),
    "L0q+C3": (("C3",), "candidate", "window"),
    "L0q+C3[lead_oracle]": (("C3",), "report_only", "oracle"),
    "G_app": (("C1", "C2"), "report_only", None),
    "NN": (None, "report_only", None),
}
CANDIDATE_MODEL = {"C1": "L0q+C1", "C2": "L0q+C2", "C3": "L0q+C3"}


def ta_protocol() -> dict:
    sel, test, gate, rule, boot, hz = (cq.TA_SELECTION, cq.TA_TEST, cq.TA_GATE, cq.TA_TEST_RULE, cq.TA_BOOT,
                                       cq.TA_HARNESS)
    cands = {c: dict(cq.TA_CANDIDATES[c]) for c in ("C1", "C2", "C3")}
    cands["C1"]["selected_on"] = "the schema tiers, chosen with all three decades (admitted)"
    cands["C2"]["selected_on"] = "the research qmodel (q03/q04), chosen on the test decades"
    cands["C3"]["selected_on"] = "the layers report, full-sample lead-lag evidence (E21)"
    return {
        "id": "T-A",
        "spec": "SIM_DESIGN §4.2 (protocol), §4.3 (the NN row), §14 P5",
        "question": "Does a registered addition to the asset layer L0 improve the oracle-macro fit of the 10 risk "
                    "assets enough to be adopted? If T-A fails or has not run, the asset layer is L0.",
        "baseline": {
            "model": "L0q",
            "definition": "scenariolab.qmodel.assets.AssetLayer: the factor round's F (data/results/"
                          "model_selection.json; monthly OLS on price returns refitted at every train end) applied to "
                          "the quarter's factor sums from the quarterly fields; tp10_x_core4 switched at the quarter's "
                          "start; mechanical assets by identity; 13-asset residual vectors in 2-quarter blocks scaled "
                          "by s_vol; 200 block-bootstrap refits of F (12-month blocks), one per path"},
        "candidates": cands,
        "candidate_fit": "per risk asset, a ridge on the candidate's terms fitted to the L0 training residuals (terms "
                         "standardized on the training quarters and saturated at +-4 sd; lambda = g x n by blocked "
                         "5-fold CV, g in {0.01, 0.03, 0.1, 0.3, 1, 3, 10}); L0's coefficients are unchanged. Several "
                         "candidates together are ONE ridge on the union of their terms "
                         "(tools/run_preregistered.JointCandidateLayer; for one candidate it equals "
                         "qmodel.assets.AssetLayer). Systematic part with C2 (states rolled by the path): the mean of "
                         "200 paths (seed 0), as qmodel's harness adapter does.",
        "c3_scoring": "C3 is a path-mode candidate (next-quarter d_ip and d_unrate). Under the harness rule (no factor "
                      "data after the window) the realized leads are used inside the window and the window's last "
                      "quarter's lead is set to the lead's training mean (the model's unconditional value); at 1Q C3 "
                      "therefore has no lead information. C3[lead_oracle] reads the realized lead of the quarter "
                      "after the window (the research evidence's form): selection period only, report-only.",
        "report_only": {
            "G_app": "L0q + C1 + C2 in one ridge: the research qmodel's G additions (tier-1 fields beyond F; states "
                     "and interactions) in the app's form. Not the research's structured Gaussian G (not ported). "
                     "The NN rule's comparator; never adopted here (its parts go through the gate one by one).",
            "NN": {"what": "port of the research qcore.MaskedNN onto the app's quarterly panel (this runner)",
                   "inputs": "[s, x * m, m]: s = 17 states + the 13 assets' last-quarter and 4-quarter returns; x = "
                             "the 52 card fields with x_oil_cause carried by d_oil_sup, + the 5 policy shocks u_* of "
                             "qmodel.policy fitted to the train end; m = the mask (1 where measured)",
                   "outputs": "13 asset log returns + y_mult, and x",
                   "config": NN_CFG,
                   "architecture": "one tanh layer of 24 units + a linear skip from [s, x * m]; standardized "
                                   "inputs saturated at +-4 sd; loss = MSE on returns + 0.5 x MSE on masked-but-"
                                   "measured x + alpha/2 ||W||^2; 8 copies of each training quarter (1 unmasked, 7 "
                                   "with random masks: iid at a random rate, or E1/E1c/E2 or E1/E2 masked); "
                                   "L-BFGS-B, 300 iterations; alpha by time-ordered validation (first 80% / last "
                                   "20% of the training quarters; full, half-masked and O-A inputs); 4 seeds",
                   "band": "the ensemble's training residual vectors drawn iid, scaled by h(s_vol) and by the "
                           "out-of-time / in-sample error ratio (clipped to [1, 2])",
                   "windows": "1Q: the mean on the origin's start states and the quarter's realized x. 4Q: "
                              "recursive as the research harness adapter (qeval.path_block): the realized x and "
                              "macro states of every quarter, the asset-derived states (P/D, CAPE, FCF yield, s_vol, "
                              "momentum) rolled with each of 300 simulated paths; s_ecyc held at the origin (the "
                              "app's y_mult is the CAPE multiple, so Δlog Ê is not an output); after the origin u_def "
                              "and u_ptax are recomputed from each path's simulated equity returns (their systematic "
                              "parts read the previous four quarters' returns; the research adapter read the "
                              "realized ones, a leak under the harness rule)"},
            "NN-res": {"run": False,
                       "reason": "its backbone is the research's structured Gaussian G with masked completion, which "
                                 "is not ported; the app's asset layer takes complete fields, so NN-res cannot be "
                                 "built as specified. Report-only (§4.3); in the research it stayed within +-0.05 of "
                                 "G and was never significant."},
            "C4": {"run": False,
                   "reason": "V-AGG needs 8 sector-group return equations; the asset layer carries 6 sectors (s_w has "
                             "8 weights). Report-only (§4.2); not implemented (P2 registered it as unavailable)."},
            "C3[lead_oracle]": "see c3_scoring",
        },
        "selection": {
            "protocol": "rolling (harness): origins at quarter starts 1990-01..2014-10 (100), refitted every January "
                        "on data to the previous December (train ends 1989-12..2013-12: 25 fits)",
            "data_cut": "the monthly panel to 2014-12 and the quarterly panel to 2014Q4 before any fit or score: no "
                        "selection step can read the test decade",
            **sel,
            "cells": "a cell holds the windows whose origin and end lie in its span (overlapping windows at 4Q)",
            "models": [m for m in TA_MODELS],
        },
        "gate": {
            "horizon_q": sel["gate_horizon_q"], "decades": list(sel["gate_decades"]), **gate,
            "statistic": "gain = mean10 R² (mean over the 10 risk assets of R² against the training-mean drift) of "
                         "L0 + candidate minus L0, on the cell's common windows; 90% interval by a moving-block "
                         "bootstrap of windows (blocks of 4, 2000 draws, the same resampled windows for both models)",
            "rule": "pass if the gain is >= +0.005 in both 1995-2004 and 2005-2014, > -0.02 in both, and its 90% "
                    "lower bound is > 0 in at least one of them",
            "4q": "computed and reported (gate_4q); not decisive",
        },
        "test": {
            "protocol": "split (harness): one fit on data to 2014-12, scored on 2015-01..2026-09",
            **test,
            "windows": {"1q": "the 47 quarters 2015Q1..2026Q3",
                        "4q": "the 11 non-overlapping calendar years 2015..2025 (origins each January); the 44 "
                              "overlapping 4Q windows are reported"},
            "models": "L0q; the union of the candidates that passed the gate (if any), as one model; the "
                      "report-only rows G_app and NN",
            "not_scored": "candidates that fail the gate; C3[lead_oracle]",
        },
        "test_rule": {
            **rule,
            "adopt_if": ["ΔR²(1Q) against L0 (47 quarters): 90% lower bound > 0 (moving blocks of 4 quarters, "
                         "2000 draws)",
                         "ΔR²(4Q) against L0 on the 11 non-overlapping windows: 90% lower bound > -0.02 (blocks of one "
                         "window = 4 quarters, 2000 draws)",
                         "80% coverage of the 1Q band (mean over the 10 risk assets) within [0.75, 0.88]"],
            "union": "the selected candidates are tested together as one model; no subset is tried after the test",
        },
        "decision_rule": ["no candidate passes the gate, or the union fails any test condition: the asset layer "
                          "stays L0, config_q.TA_DEFAULT = ()",
                          "otherwise config_q.TA_DEFAULT = the selected candidates (id order); fields outside F "
                          "reach assets only through the factor layer until then (§4.2 item 4)"],
        "nn_reentry_rule": "report-only (§4.3, D4, U-10). The NN qualifies for re-entry only if it beats both G_app "
                           "and L0 with a 90% lower bound > 0 at 1Q and at 4Q in both gate decades and in the test. "
                           "Qualifying does not adopt it: the asset layer has no NN path; adoption would need a new "
                           "decision and record.",
        "harness": {**hz, "metrics": "scenariolab.evaluation.harness (R² vs the training-mean drift, its 90% block "
                                     "interval, CRPSS against the drift's distribution, 80% coverage, PIT)",
                    "price_returns": "cumulative log price returns of the 10 risk assets; mechanical assets are not "
                                     "scored"},
        "bootstrap": boot,
        "seeds": {"conditional_paths": hz["seed"], "asset_layer_param_refits": hz["fit_seed"],
                  "candidate_systematic_mean": 0, "paired_bootstrap": cq.PREREG_SEED,
                  "metrics_bootstrap": boot["metrics_seed"],
                  "nn": {"fit": NN_CFG["seed"], "validation_masks": NN_CFG["val_mask_seed"],
                         "window_mean": "7 + 12 x year + month of the origin", "band": "the harness window seed"}},
        "disclosure": [
            "2015-26 has been seen by every earlier round: the factor round's test was used for its choice, the "
            "schema tiers used it, and the qmodel scored its variants on it. This is a pre-registered re-test, not a "
            "fresh holdout. The only clean holdout is the prospective ledger from 2026Q4 (§9).",
            "F's 1995-2004 was its validation period (GZ was selected with 1995-2004 as validation): L0 itself is "
            "not clean in that decade.",
            "Before this record no implementation round scored a T-A candidate: P2's tests compute C1-C3 "
            "predictions on 2015Q1-2016Q4 only to check plumbing. L0's own split and rolling scores, including "
            "2015-26, were computed by P2 (scratch x13) and P2 reported the quarterly L0's gap to the research L0.",
            "The NN port is new: the research NN's scores (§4.3) are from the research panel; this run is the "
            "app's first NN score.",
            "Before freezing, the runner's plumbing was checked on data cut at the end of 2014 (selection windows "
            "2012-14, a stand-in test year 2014, a forced union) and on 1999-2000 (joint layer = AssetLayer for one "
            "candidate; the C3 lead rule; NN fit and finite predictions); no score was printed or read.",
        ],
        "interpretations": [
            "The gate is evaluated at 1Q (the schema's tier rule is a same-quarter increment); 4Q is reported.",
            "Selection origins 1990Q1-1994Q4 are scored and reported (cells 1990-1994 and 1990-2014) but are not "
            "in the gate, which names two decades.",
            "The test's 4Q statistic uses the 11 non-overlapping calendar-year windows; its bootstrap resamples "
            "whole windows.",
            "Coverage at 1Q = the mean over the 10 risk assets of the share of the 47 test quarters inside the "
            "10-90% band.",
            "C3 under the harness rule: see c3_scoring.",
            "A union of selected candidates is fitted jointly (one ridge).",
            "G is G_app (L0 + C1 + C2), not the research's structured Gaussian.",
            "CMA 'historical' for every model (walk-forward drift; R² against the training-mean drift).",
        ],
    }


# ================================================================ T-A: the run

def _metric_cells(m: pd.DataFrame, model: str, protocol: str) -> dict:
    keep = ("n", "n_indep", "r2", "r2_lo90", "r2_hi90", "r2_median", "crpss", "cov80", "pit_mad", "corr", "mae",
            "bias")
    out = {}
    if m.empty:
        return out
    sub = m[(m.model == model) & (m.protocol == protocol) & (m.asset == "mean10")]
    for _, r in sub.iterrows():
        out.setdefault(str(r["group"]), {})[f"{int(r['h_q'])}q"] = {k: r[k] for k in keep if k in r}
    return out


def _asset_r2(m: pd.DataFrame, model: str, protocol: str) -> dict:
    out = {}
    if m.empty:
        return out
    sub = m[(m.model == model) & (m.protocol == protocol) & (m.asset != "mean10")]
    for _, r in sub.iterrows():
        out.setdefault(str(r["group"]), {}).setdefault(f"{int(r['h_q'])}q", {})[r["asset"]] = r["r2"]
    return out


def _diff_cells(d: pd.DataFrame) -> dict:
    out = {}
    for _, r in d.iterrows():
        out.setdefault(str(r["group"]), {})[f"{int(r['h_q'])}q"] = {
            "diff": r["diff_r2"], "lo90": r["lo90"], "hi90": r["hi90"], "n": int(r["n"]),
            "n_indep": int(r["n_indep"])}
    return out


def gate_result(diffs: dict, proto: dict, h: int | None = None) -> dict:
    g = proto["gate"]
    h = int(h or g["horizon_q"])
    cells = {dec: diffs.get(dec, {}).get(f"{h}q") for dec in g["decades"]}
    if any(v is None or v["diff"] is None or v["lo90"] is None for v in cells.values()):
        return {"pass": False, "horizon_q": h, "reason": "a decade has no paired windows", "cells": cells}
    cond = {"min_gain_both": all(v["diff"] >= g["min_gain"] for v in cells.values()),
            "above_floor_both": all(v["diff"] > g["floor"] for v in cells.values()),
            "lower_bound_one": any(v["lo90"] > g["lb_above"] for v in cells.values())}
    return {"pass": all(cond.values()), "horizon_q": h, "conditions": cond, "cells": cells}


def _nonoverlap(paths: pd.DataFrame, h: int = 4) -> pd.DataFrame:
    """The h-quarter windows that start each January (non-overlapping calendar years); other horizons dropped."""
    o = pd.to_datetime(paths["origin"])
    return paths[(paths["h_q"] == h) & (o.dt.month == 1)].copy()


def _lb_ok(cell: dict | None, above: float) -> bool:
    return bool(cell is not None and cell.get("lo90") is not None and cell["lo90"] > above)


def run_ta(protocol: dict, *, P: pd.DataFrame | None = None, Q: pd.DataFrame | None = None, out_dir: Path | None = None,
           tag: str = "TA", verbose: bool = True, nn_cfg: dict | None = None, _force_selected=None) -> dict:
    """Run T-A as `protocol` says (the frozen record's protocol). Returns the result to append. _force_selected
    (smoke runs only) replaces the gate's selection so the union branch runs; a recorded run never passes it."""
    eh = _harness()
    from scenariolab.qmodel import panel as qpanel
    t_all = time.time()
    P = eh.load_panel(valuation=True) if P is None else P
    Q = qpanel.load_panel() if Q is None else Q
    sel, test, hz, boot = protocol["selection"], protocol["test"], protocol["harness"], protocol["bootstrap"]
    out_dir = Path(out_dir or cq.PREREG_RESULTS_DIR)
    seconds, fit_log = {}, []

    def say(*a):
        if verbose:
            print(*a, flush=True)

    def model_fn(name: str, Qx: pd.DataFrame):
        cands, _, lead = TA_MODELS[name]
        if name == "NN":
            return nn_model_fn(Qx, name=name, cfg=nn_cfg, fit_log=fit_log)
        return joint_model_fn(Qx, cands, name=name, lead_rule=lead or "window", cma=hz["cma"],
                              n_boot=int(hz["n_boot_params"]), seed=int(hz["fit_seed"]), fit_log=fit_log)

    common = {"band": True, "n_paths": int(hz["n_paths"]), "seed": int(hz["seed"]), "crps": True}
    # ---------------------------------------------------------------- selection (data cut at the end of 2014)
    end_m = pd.Timestamp(sel["data_end"])
    Ps, Qs = P.loc[:end_m], Q.loc[:pd.Period(end_m, "Q")]
    spans = {g: tuple(v) for g, v in sel["spans"].items()}
    parts = []
    for name in sel["models"]:
        t0 = time.time()
        parts.append(eh.conditional_paths(model_fn(name, Qs), Ps, protocol="rolling",
                                          horizons_q=tuple(sel["horizons_q"]), roll_first=sel["roll_first"],
                                          roll_refit=int(sel["refit_months"]), roll_last=sel["roll_last"], **common))
        seconds[f"selection:{name}"] = round(time.time() - t0, 1)
        say(f"  selection {name}: {seconds[f'selection:{name}']} s")
    paths_sel = pd.concat(parts, ignore_index=True)
    m_sel = eh.metrics(paths_sel, spans=spans, n_boot=int(boot["metrics_n_boot"]), seed=int(boot["metrics_seed"]))
    sel_cells = {name: _metric_cells(m_sel, name, "rolling") for name in sel["models"]}
    sel_diff = {}
    for name in sel["models"]:
        if name == "L0q":
            continue
        d = eh.paired_diff(paths_sel, "L0q", name, protocol="rolling", spans=spans, horizons_q=tuple(sel["horizons_q"]),
                           n_boot=int(boot["n_boot"]), block=int(boot["block_1q"]), seed=cq.PREREG_SEED)
        sel_diff[name] = _diff_cells(d)
    nn_vs_g_sel = _diff_cells(eh.paired_diff(paths_sel, "G_app", "NN", protocol="rolling", spans=spans,
                                             horizons_q=tuple(sel["horizons_q"]), n_boot=int(boot["n_boot"]),
                                             block=int(boot["block_1q"]), seed=cq.PREREG_SEED))
    gates = {}
    for cid, name in CANDIDATE_MODEL.items():
        gates[cid] = {**gate_result(sel_diff[name], protocol),
                      "gate_4q": gate_result(sel_diff[name], protocol, h=4)}
    selected = [cid for cid in ("C1", "C2", "C3") if gates[cid]["pass"]]
    if _force_selected is not None:
        selected = list(_force_selected)
    say(f"  selection done ({len(selected)} candidate(s) passed)")
    # ---------------------------------------------------------------- test (one fit to 2014-12, 2015-01..2026-09)
    union = "L0q+" + "+".join(selected) if selected else None
    if union:
        TA_MODELS[union] = (tuple(selected), "test_union", "window")
    test_models = ["L0q"] + ([union] if union else []) + ["G_app", "NN"]
    split = {test["group"]: (test["train_end"], test["first"], test["last"])}
    parts = []
    for name in test_models:
        t0 = time.time()
        parts.append(eh.conditional_paths(model_fn(name, Q), P, protocol="split", horizons_q=tuple(test["horizons_q"]),
                                          splits=split, **common))
        seconds[f"test:{name}"] = round(time.time() - t0, 1)
        say(f"  test {name}: {seconds[f'test:{name}']} s")
    paths_test = pd.concat(parts, ignore_index=True)
    paths_nov = _nonoverlap(paths_test, 4)
    m_test = eh.metrics(paths_test, n_boot=int(boot["metrics_n_boot"]), seed=int(boot["metrics_seed"]))
    m_nov = eh.metrics(paths_nov, n_boot=int(boot["metrics_n_boot"]), seed=int(boot["metrics_seed"]))
    test_cells = {name: {"overlapping": _metric_cells(m_test, name, "split"),
                         "nonoverlap_4q": _metric_cells(m_nov, name, "split")} for name in test_models}

    def tdiff(a, b):
        d1 = _diff_cells(eh.paired_diff(paths_test, a, b, protocol="split", horizons_q=(1, 4),
                                        n_boot=int(boot["n_boot"]), block=int(boot["block_1q"]), seed=cq.PREREG_SEED))
        d4 = _diff_cells(eh.paired_diff(paths_nov, a, b, protocol="split", horizons_q=(4,), n_boot=int(boot["n_boot"]),
                                        block=int(boot["block_4q_nonoverlap"]), seed=cq.PREREG_SEED))
        g = test["group"]
        return {"1q": d1.get(g, {}).get("1q"), "4q_overlapping": d1.get(g, {}).get("4q"),
                "4q_nonoverlap": d4.get(g, {}).get("4q")}

    test_diff = {name: tdiff("L0q", name) for name in test_models if name != "L0q"}
    nn_vs_g_test = tdiff("G_app", "NN")
    # ---------------------------------------------------------------- decision
    rule = protocol["test_rule"]
    if union:
        td = test_diff[union]
        cov = test_cells[union]["overlapping"].get(test["group"], {}).get("1q", {}).get("cov80")
        cond = {"lb_1q": _lb_ok(td["1q"], rule["lb_1q_above"]),
                "lb_4q": _lb_ok(td["4q_nonoverlap"], rule["lb_4q_above"]),
                "cov80_1q": bool(cov is not None and rule["cov80_1q"][0] <= cov <= rule["cov80_1q"][1])}
        adopted = list(selected) if all(cond.values()) else []
        test_eval = {"model": union, "conditions": cond, "cov80_1q": cov, "diff": td, "adopted": bool(adopted)}
    else:
        adopted = []
        test_eval = {"model": None, "conditions": None, "adopted": False,
                     "note": "no candidate passed the gate; nothing was tested for adoption"}
    decision = {"asset_layer": "L0" if not adopted else "L0 + " + " + ".join(adopted), "ta_default": adopted,
                "selected": selected, "rule": protocol["decision_rule"]}
    l0t = test_cells["L0q"]
    counts = {"n_1q": (l0t["overlapping"].get(test["group"], {}).get("1q") or {}).get("n"),
              "n_4q_nonoverlap": (l0t["nonoverlap_4q"].get(test["group"], {}).get("4q") or {}).get("n")}
    counts["as_registered"] = bool(counts["n_1q"] == test.get("n_1q")
                                   and counts["n_4q_nonoverlap"] == test.get("n_4q_nonoverlap"))
    # ---------------------------------------------------------------- the NN row (report-only)
    reentry = {}
    for comp, sd_, td_ in (("L0q", sel_diff["NN"], test_diff["NN"]), ("G_app", nn_vs_g_sel, nn_vs_g_test)):
        for dec in protocol["gate"]["decades"]:
            for h in (1, 4):
                reentry[f"vs_{comp}:{dec}:{h}q"] = _lb_ok(sd_.get(dec, {}).get(f"{h}q"), 0.0)
        reentry[f"vs_{comp}:test:1q"] = _lb_ok(td_["1q"], 0.0)
        reentry[f"vs_{comp}:test:4q"] = _lb_ok(td_["4q_nonoverlap"], 0.0)
    nn_fits = [f for f in fit_log if f.get("model") == "NN"]
    nn_row = {"model": "NN", "role": "report-only (§4.3)", "adopted": False,
              "selection": sel_cells["NN"], "test": test_cells["NN"],
              "vs_L0": {"selection": sel_diff["NN"], "test": test_diff["NN"]},
              "vs_G_app": {"selection": nn_vs_g_sel, "test": nn_vs_g_test},
              "reentry_conditions": reentry, "qualifies_for_reentry": bool(all(reentry.values())),
              "fits": nn_fits,
              "alpha_by_fit": {f["train_end"]: f["alpha"] for f in nn_fits}}
    # ---------------------------------------------------------------- derived rows
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, df in (("selection_paths", paths_sel), ("test_paths", paths_test)):
        p = out_dir / f"{tag}_{key}.csv.gz"
        df.to_csv(p, index=False)
        files[key] = {"path": rel(p), "rows": int(len(df)), "sha256": sha256_file(p)}
    l0_sel = sel_cells["L0q"]
    result = {
        "selection": {"cells": sel_cells, "diff_vs_L0": sel_diff, "nn_vs_G_app": nn_vs_g_sel, "gate": gates,
                      "selected": selected},
        "test": {"models": test_models, "cells": test_cells, "diff_vs_L0": test_diff, "nn_vs_G_app": nn_vs_g_test,
                 "evaluation": test_eval,
                 "per_asset_r2": {n: _asset_r2(m_test, n, "split") for n in test_models}},
        "decision": decision,
        "test_window_counts": counts,
        "nn_row": nn_row,
        "not_run": {k: v for k, v in protocol["report_only"].items() if isinstance(v, dict) and v.get("run") is False},
        "l0_reference": {"selection": l0_sel, "test": test_cells["L0q"]},
        "fits": [f for f in fit_log if f.get("model") != "NN"],
        "files": files,
        "seconds": {**seconds, "total": round(time.time() - t_all, 1)},
        "disclosure": protocol["disclosure"],
    }
    if union:
        TA_MODELS.pop(union, None)
    return jsonable(result)


# ================================================================ T-C

def tc_protocol() -> dict:
    """T-C as re-frozen by the fix round (supersedes 2026-11-02_TC.json: same question, data, seed, patterns, w grid,
    selection rule, regime rule and gate; the scoring, the information set and the saturation rule are new)."""
    return {
        "id": "T-C",
        "revision": 2,
        "spec": "SIM_DESIGN §3.2, §3.5, §5.7, §5.8; §14 P5",
        "question": "Choose the completion weight w and decide the regime-dependent covariance on validation; score "
                    "the completion of unstated factors once on the test; gate: where the model cannot infer, it "
                    "stays near unconditional.",
        "implementation": "scenariolab.qmodel.factor_checks.tc(Q, P, seed, scoring='exact'), called once by this runner "
                          "on the monthly panel P and the quarterly panel Q; qmodel.json's stored T_C is not read for "
                          "the decision",
        "validation": {"origins": f"{cq.TC_VALIDATION[0]}..{cq.TC_VALIDATION[1]} (80 quarters)",
                       "refits": "yearly, each January on data to the previous December (fit at (year-1)Q4, "
                                 "anchors fitted to the train end, §5.5)"},
        "test": {"fit_end": cq.TC_TEST_END, "origins": f"{pd.Period(cq.TC_TEST_END, 'Q') + 1}..{cq.TC_TEST_LAST} "
                                                       "(47 quarters)"},
        "mode": "step mode (H = 1): the Gaussian conditional moments of the quarter given the origin's state and the "
                "stated fields (factors.conditional_moments)",
        "scoring": "exact",
        "scoring_text": "every x field is a monotone transform of one linear combination of the stacked quarter, so its "
                        "predictive mean, CRPS and 80% interval are computed exactly (factor_checks.step_moments: "
                        "Gaussian closed forms, Gauss-Hermite for the bounded scales, censored-normal means at the "
                        "floors, the pinball integral on a fixed normal-quantile grid); no Monte Carlo, so the "
                        "conditional and unconditional sets differ only by the statements and every gate cell's MC "
                        "standard error is 0",
        "information": "the origin's state from data published by its start only (factors.initial_state 'published', "
                       "§3.2: the previous quarter's CPI, IP, unemployment, NIPA, Z.1, profits and the budget are "
                       "imputed from the joint distribution given what was published)",
        "patterns": {"a": "a random 50% of the factor layer's x fields stated (rng([seed, year, quarter]))",
                     "b": "only the E0 and E0p fields stated (narrative-like)",
                     "c": "only the financial-conditions fields stated (card fc_role required_one or optional)"},
        "stated_values": "the realized values, saturated as §3.5 writes it (the gap to the conditional mean at +-4 "
                         "training sd, factors.saturation); d_ff is not stated in zero-bound quarters (the shadow rate "
                         "is not observed at the floor)",
        "metrics": "per unstated field: completion R² against the model's unconditional (state-only) prediction, "
                   "CRPS, 80% coverage (the exact 10-90% interval)",
        "w_grid": list(cq.COMPLETION_W_GRID),
        "selection_w": "w* = argmin over the grid of the mean CRPS relative to the unconditional CRPS, pattern (b), "
                       "E1 / E1c / E2 fields, pooled covariance, validation origins",
        "regime_cov": {"n0": cq.REGIME_N0, "regimes": "s_zlb, s_core4",
                       "adopt_if": "at w*, pattern (b): the E1-class mean relative CRPS in ZLB quarters is lower with "
                                   "the regime covariance than without, and its change in every other cell (pattern "
                                   "a, c, and b outside the ZLB; ZLB and normal quarters) is at most +0.01"},
        "gate": "pattern (b) on the test: every E1-class field's completion R² >= -0.02 (exact; MC se 0)",
        "seed": cq.TF_SEED, "draws": None,
        "application": "This run records T-C's selection and gate. It does not change config_q.COMPLETION_W or "
                       "REGIME_COV: path mode (qmodel.factors.simulate) would follow them while step mode "
                       "(qmodel.rollout.Stepper) implements neither, so switching them breaks the step / path "
                       "consistency of §3.7. The lead decides after the run.",
        "supersedes_because": [
            "the frozen record's gate was decided by Monte Carlo noise: R² on means of 1,000 draws per origin with "
            "independent conditional and unconditional streams (re-scored on 8 draw seeds the gate passed in 3; the "
            "probability that the linear E1 satellites pass was about 0.13)",
            "its initial states used the previous quarter's realized values, which the panel's own release lags mark "
            "as unpublished at the origin (38 of 52 stacked variables at a January origin)",
            "its saturation band was ±4 sd of a 400-path Gaussian prior with the t-weighted Sigma, narrower than §3.5's "
            "±4 training sd",
        ],
        "disclosure": [
            "Seen before this record was frozen: the frozen record's run (2026-11-02_TC.json: w* 0.25, regime "
            "covariance not adopted, gate failed with d_tp10 at -0.029) and the statistics review's re-scoring of its "
            "test block: exact Gaussian means under the frozen information gave d_tp10 -0.0206 and d_bdlev -0.0304 "
            "(both below -0.02); hiding only the 18 unpublished satellites gave d_tp10 -0.0048, d_cpbill 0.0043, "
            "d_hpi 0.0222. This record's combination (exact scoring, every unpublished value imputed, §3.5 "
            "saturation) had not been run on the test before it was frozen; its plumbing was checked on validation "
            "origins only.",
            "The test decade 2015-26 has been seen by earlier rounds (§4.2 disclosure).",
            "Nothing else changed: data files, seed, patterns, w grid, selection and regime rules and the gate are the "
            "frozen record's. The factor layer's code changed in the same fix round (the bootstrap pool is centred, "
            "which step-mode Gaussian scoring does not use; the rarity statistic; the QTable reader).",
        ],
        "interpretations": [
            "'E1 field' in the gate = card endogeneity class E1 (financial markets); the w selection uses E1, E1c "
            "and E2 (the classes §3.5 shrinks).",
            "Completion is scored in step mode (one quarter), as the implementation does.",
        ],
    }


def max_abs_diff(a, b) -> tuple[float, int]:
    """(largest |a - b| over the numeric leaves both share, number of structural mismatches)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out, mism = 0.0, len(set(a) ^ set(b))
        for k in set(a) & set(b):
            d, m = max_abs_diff(a[k], b[k])
            out, mism = max(out, d), mism + m
        return out, mism
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 0.0, 1
        out, mism = 0.0, 0
        for x, y in zip(a, b):
            d, m = max_abs_diff(x, y)
            out, mism = max(out, d), mism + m
        return out, mism
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0, int(a != b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)), 0
    return 0.0, int(a != b)


def run_tc(protocol: dict, verbose: bool = True) -> dict:
    from scenariolab.data import panel as mpanel
    from scenariolab.qmodel import factor_checks as fc
    from scenariolab.qmodel import panel as qpanel
    t0 = time.time()
    P = mpanel.load_panel()
    Q = qpanel.load_panel()
    scoring = protocol.get("scoring", "mc")         # the 2026-11-02 record predates the exact scoring
    res = jsonable(fc.tc(Q, P, seed=int(protocol["seed"]), n=int(protocol.get("draws") or cq.TC_DRAWS),
                         scoring=scoring))
    sel = res["selection"]
    decision = {"completion_w": sel["completion_w"], "regime_cov_adopted": sel["regime_cov"]["adopted"],
                "gate_pass": res["gate"]["pass"], "scoring": scoring, "gate_mc_se": res["gate"].get("mc_se"),
                "gate_failures": {f: v for f, v in res["gate"]["E1_r2"].items() if v is not None and v < -0.02},
                "applied_to_config_q": False, "application": protocol["application"],
                "config_q_now": {"COMPLETION_W": cq.COMPLETION_W, "REGIME_COV": cq.REGIME_COV}}
    dev = {}
    if cq.QMODEL_RESULTS_PATH.exists():
        d = json.loads(cq.QMODEL_RESULTS_PATH.read_text(encoding="utf-8")).get("factor_layer", {})
        old = d.get("T_C")
        if old:
            diff, mism = max_abs_diff(old, res)
            dev = {"created": d.get("created"), "note": "qmodel.json rounds to 6 decimals (jsonio)",
                   "max_abs_numeric_diff": diff, "structural_mismatches": mism,
                   "reproduced_to_1e-6": bool(mism == 0 and diff <= 1.5e-6),
                   "w_same": old["selection"]["completion_w"] == sel["completion_w"],
                   "regime_cov_same": old["selection"]["regime_cov"]["adopted"] == sel["regime_cov"]["adopted"],
                   "gate_same": old["gate"]["pass"] == res["gate"]["pass"]}
    if verbose:
        print(f"  T-C: {round(time.time() - t0, 1)} s", flush=True)
    return {"tc": res, "decision": decision, "development_run": dev, "seconds": round(time.time() - t0, 1)}


# ================================================================ T-S

TS_EPISODES = {  # §8.2: the clean set (>= 56 training quarters); values from [d02]
    "E11": ("1990Q3", "1991Q4", 70, False), "N13": ("1987Q3", "1988Q4", 58, True),
    "E07": ("1994Q1", "1995Q2", 84, False), "E02": ("1995Q3", "1997Q4", 90, False),
    "N14": ("1998Q1", "2000Q4", 100, False), "E12": ("2001Q1", "2002Q4", 112, True),
    "E01": ("2004Q3", "2006Q4", 126, False), "E05": ("2007Q3", "2009Q2", 138, True),
    "N15": ("2010Q2", "2012Q4", 149, False), "E04": ("2014Q3", "2015Q4", 166, False),
    "E09": ("2017Q1", "2019Q4", 176, False), "E08": ("2020Q1", "2021Q2", 188, True),
    "E10": ("2022Q1", "2023Q4", 196, True), "N18": ("2024Q1", "2026Q2", 204, False),
}
TS_SHORT = {"E06": "1973Q4-1975Q2", "E03": "1979Q4-1982Q2", "N16": "1977Q1-1979Q3", "N17": "1983Q1-1984Q4"}


def ts_protocol(date: str) -> dict:
    mg = cq.TS_MARGINS
    return {
        "id": "T-S",
        "spec": "SIM_DESIGN §4.6 (the gate), §8 (the narrative test), §12 step 7, §14 P9",
        "question": "Does the quarterly stack become the default for the 13 scenarios, the forecast tab, portfolio "
                    "and withdrawal? Until it passes, the monthly engine stays the default and the quarterly stack is "
                    f"labelled {cq.TRIAL_LABEL} (U-1).",
        "evaluated_by": "P9, from this record, the T-A record of the same date and P7's narrative-test records",
        "episodes": {"set": "the clean set (§8.2: at least 56 training quarters)",
                     "labels": "SIM_DESIGN §8.2's public labels, not the narrative registry's shuffled codes (the "
                               "sealed key in data/evaluation/key.json stays sealed)",
                     "clean": {k: {"window": f"{a}..{b}", "train_quarters": n, "bust": bust}
                               for k, (a, b, n, bust) in TS_EPISODES.items()},
                     "excluded": {"short_training": TS_SHORT, "why": "reported apart, never pooled (§8.2)"}},
        "engines": {"Q": "the quarterly stack in path mode, with the asset layer the T-A record decides (L0 unless "
                         "it adopts candidates) and the factor layer as P9 evaluates it",
                    "E": "the monthly engine running the selected model F (data/results/model_selection.json), not "
                         "the base model; it receives the same statements through qtable.to_phase_spec (§7.4)"},
        "arms": {"oracle": {"O-A": "realized E0, E0p (raw) and x_oil_cause",
                            "O-B": "O-A + realized E1c + realized E1 (financial markets)",
                            "O-full": "O-B + realized E2 = every x field"},
                 "narrative": {"N1": "macro and policy narrative", "N2": "N1 + financial conditions",
                               "N3": "N1 without digits"},
                 "statements": "both engines receive identical statements: the arm's realized fields, or the same "
                               "translated QTable, converted for E by qtable.to_phase_spec"},
        "conditions": {
            "1": {"rule": "T-F passes: every check of qmodel.json['factor_layer']['T_F']['summary'] is true, "
                          "recomputed (build step qmodel) at the code P9 evaluates"},
            "2": {"rule": "oracle arms O-A, O-B and O-full: Q - E equity9 MAE has a 90% upper bound below +0.02, at "
                          "4Q and at the window end (6 comparisons, all must hold)", "margin": mg["oracle_ub"]},
            "3": {"rule": "calibration: the mean |PIT - 0.5| of Q is at most E's + 0.02 (pooled over the six arms, "
                          "the 9 equity-like assets, the clean episodes, 4Q and the window end)",
                  "margin": mg["pit_mad_excess"]},
            "4": {"rule": "narrative arms N1, N2 and N3 on identical translations: Q - E equity9 MAE has a 90% "
                          "upper bound below +0.03, at 4Q and at the window end (6 comparisons)",
                  "margin": mg["narrative_ub"]},
        },
        "measures": {"equity9_mae": "per episode, the mean over the 9 equity-like assets of |median simulated "
                                    "cumulative log price return - realized| from the episode's first quarter",
                     "horizons": ["4Q", "window end"],
                     "difference": "Q - E per episode; the mean over episodes",
                     "upper_bound": "the 95th percentile of 2000 bootstrap means over episodes (seed "
                                    f"{cq.PREREG_SEED}): a 90% two-sided interval's upper end",
                     "pit": "the share of a model's simulated cumulative returns below the realized value"},
        "bootstrap": cq.TS_BOOT, "paths": cq.TS_PATHS, "seed": cq.PREREG_SEED,
        "cma": {"decisive": "historical (both engines: walk-forward drift)", "reported": "no_rerate (§4.4)"},
        "translations": "condition 4 counts only a narrative-test run on the v2 narratives (blind protocol) with a "
                        "real backend that the user started by hand (§8.5, U-8); mock runs and the research round's "
                        "stand-in translations (研究轮代译) do not count",
        "decision_rule": ["all four conditions hold: the defaults switch (P9; quarterly sets for portfolio and "
                          "withdrawal)",
                          "any condition fails or cannot be evaluated: nothing switches, the evaluation tab shows "
                          "which condition failed with its numbers"],
        "margin_note": "non-inferiority with a tight margin, not superiority (§4.6): the research qmodel would fail "
                       "(oracle upper bound +0.021; narratives +0.056 to +0.067 with every lower bound above 0)",
        "interpretations": [
            "Condition 4's horizons are those of condition 2 (4Q and the window end).",
            "Condition 3 pools the six arms; E's PIT uses its own band.",
            "The decisive CMA arm is 'historical' for both engines; 'no_rerate' is reported.",
            "E's oracle arms receive the realized fields as hard statements through the same conversion as the "
            "narrative arms.",
        ],
        "frozen_date": date,
    }


def ts_status(protocol: dict) -> dict:
    """T-S cannot be evaluated before P7: record what exists and what is missing (condition 1 as a snapshot)."""
    snap = {"available": False}
    p = cq.QMODEL_RESULTS_PATH
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8")).get("factor_layer", {})
        s = (d.get("T_F") or {}).get("summary") or {}
        snap = {"available": True, "file": rel(p), "sha256": sha256_file(p), "created": d.get("created"),
                "checks": len(s), "passed": int(sum(bool(v) for v in s.values())),
                "failed": sorted(k for k, v in s.items() if not v),
                "condition_1_would_hold": bool(s) and all(bool(v) for v in s.values())}
    narr = [rel(x) for x in (config.RESULTS_DIR / "narrative", config.RESULTS_DIR / "evaluation.json") if x.exists()]
    return {"evaluable": False, "switch": False,
            "missing": ["P7: the narrative test (episode registry and sealed key; the oracle arms O-A, O-B, O-full on "
                        "both engines; the narrative arms N1-N3 on identical translations; PIT) has not been built "
                        "or run",
                        "P6: qtable.to_phase_spec (E's statements) does not exist yet",
                        "a real-backend narrative run on the v2 narratives, started by the user (condition 4)"],
            "narrative_records_found": narr,
            "conditions": {"1": {"status": "snapshot only (evaluated by P9)", "snapshot": snap},
                           "2": {"status": "not evaluable (needs P7)"}, "3": {"status": "not evaluable (needs P7)"},
                           "4": {"status": "not evaluable (needs P6, P7 and a real run)"}},
            "consequence": "the monthly engine stays the default; the quarterly stack serves the lab and the agent "
                           f"tree, labelled {cq.TRIAL_LABEL}",
            "zh": "切换关 T-S 现在不能评估：叙事检验（P7）还没有建成，也没有运行过。月度引擎仍是默认，季度模型标“试运行”。"
                  "条件 1（T-F）只记录当前快照，正式评估在 P9。"}


# ================================================================ smoke run (before freezing)

def smoke() -> dict:
    """Run every code path of the T-A, T-C-free runner on data cut at the end of 2014 (selection 2012-14, a stand-in
    test 2014), writing records to a temporary directory. Prints run times and the shape of the result, no score."""
    eh = _harness()
    from scenariolab.qmodel import panel as qpanel
    P = eh.load_panel(valuation=True).loc[:"2014-12"]
    Q = qpanel.load_panel().loc[:"2014Q4"]
    proto = ta_protocol()
    proto["selection"].update({"roll_first": "2012-01", "roll_last": "2014-10",
                               "spans": {"2012": ("2012-01", "2012-12"), "2013-2014": ("2013-01", "2014-12"),
                                         "2012-2014": ("2012-01", "2014-12")}})
    proto["gate"]["decades"] = ["2012", "2013-2014"]
    proto["test"].update({"train_end": "2013-12", "first": "2014-01", "last": "2014-12", "group": "smoke"})
    proto["harness"]["n_paths"] = 100
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        res = run_ta(proto, P=P, Q=Q, out_dir=Path(tmp), tag="smoke", verbose=False)
        forced = run_ta(proto, P=P, Q=Q, out_dir=Path(tmp), tag="smoke_union", verbose=False,
                        _force_selected=("C1", "C3"))
        union_ok = (forced["test"]["evaluation"]["model"] == "L0q+C1+C3"
                    and set(forced["test"]["evaluation"]["conditions"]) == {"lb_1q", "lb_4q", "cov80_1q"}
                    and isinstance(forced["decision"]["ta_default"], list) and "L0q+C1+C3" not in TA_MODELS)
        # the record lifecycle on a throwaway record
        d = Path(tmp) / "rec"
        paths = freeze("1999-01-01", d, {"TA": proto, "TC": tc_protocol(), "TS": ts_protocol("1999-01-01")})
        before = paths["TA"].read_bytes()
        append_result(paths["TA"], "run", res, now(), {"matches": True})
        rec = load_record(paths["TA"])
        assert frozen_copy(rec).encode("utf-8") == before
        assert check_record(rec) == []
        try:
            append_result(paths["TA"], "run", res, now(), {"matches": True})
            raise AssertionError("a second run was accepted")
        except SystemExit:
            pass
        n_files = len(list(Path(tmp).glob("smoke_*.csv.gz")))
        out = {"seconds": res["seconds"], "result_keys": sorted(res), "n_files": n_files,
               "json_bytes": len(dumps_record(res)), "lifecycle_ok": True, "wall": round(time.time() - t0, 1),
               "nn_fits": len(res["nn_row"]["fits"]), "union_branch_ok": bool(union_ok),
               "union_seconds": forced["seconds"]["total"],
               "finite_selection_cells": int(sum(v.get("r2") is not None for m in res["selection"]["cells"].values()
                                                 for g in m.values() for v in g.values())),
               "finite_test_cells": int(sum(v.get("r2") is not None for m in res["test"]["cells"].values()
                                            for kind in m.values() for g in kind.values() for v in g.values()))}
    return out


# ================================================================ CLI

def _date(a) -> str:
    return a.date or datetime.now().strftime("%Y-%m-%d")


def run_record(path: Path) -> dict:
    """Run a runnable record once (refusing a broken record, inputs that differ from the frozen ones, or a second
    run) and append the result."""
    rec = load_record(path)
    probs = check_record(rec)
    if probs:
        raise SystemExit(f"记录不完整：{probs}")
    rid = rec["record"]
    started = now()
    bad = verify_inputs(rec["frozen"]["inputs"])
    if bad:
        raise SystemExit(f"输入与冻结时不同，拒绝运行（{len(bad)} 个文件）：{bad[:5]}")
    if any(e.get("kind") == "run" for e in rec["results"]):
        raise SystemExit(f"{rid} 已经运行过一次")
    check = {"matches": True, "code_digest": rec["frozen"]["inputs"]["code"]["digest"],
             "data_digest": rec["frozen"]["inputs"]["data"]["digest"],
             "card_digest": rec["frozen"]["inputs"]["card"]["digest"], "environment": environment()}
    proto = rec["frozen"]["protocol"]
    result = run_ta(proto, tag=Path(path).stem) if rid == "TA" else run_tc(proto)
    return append_result(path, "run", result, started, check)


def run(ctx=None, reproduce: bool = True, **_) -> dict:
    """Build step `evaluation` (opt-in): every record's integrity; for each run record, whether today's inputs are the
    frozen ones and, if they are, a re-execution compared with its stored result (reproduced or not, without
    appending anything); a record whose inputs differ is reproducible only on its frozen code (git checkout of the
    commit it names) and is reported with the files that differ. Writes data/results/evaluation.json."""
    del ctx
    t0 = time.time()
    out = {"records": {}, "label": cq.TRIAL_LABEL, "zh": "预注册记录：完整性、输入是否与冻结时相同、能否复现"}
    for p in sorted(Path(cq.PREREG_DIR).glob("*_T?.json")):
        rec = load_record(p)
        r = {"record": rec.get("record"), "problems": check_record(rec),
             "results": [(e["n"], e["kind"]) for e in rec.get("results", [])],
             "supersedes": (rec.get("frozen") or {}).get("supersedes", {}).get("record"),
             "code_commit": (rec.get("frozen") or {}).get("environment", {}).get("code_commit")}
        runs = [e for e in rec.get("results", []) if e.get("kind") == "run"]
        if rec.get("record") in ("TA", "TC") and runs:
            bad = verify_inputs(rec["frozen"]["inputs"])
            r["inputs_differ"] = [f"{b['group']}:{b['file']}:{b.get('status', 'changed')}" for b in bad]
            if bad:
                r["reproduction"] = {"status": "not run", "reason": "the inputs differ from the frozen ones: reproduce "
                                     "on the frozen code (git checkout of its commit)"}
            elif reproduce:
                res = jsonable(run_ta(rec["frozen"]["protocol"], tag=p.stem + "_reproduce", verbose=False)
                               if rec["record"] == "TA" else run_tc(rec["frozen"]["protocol"], verbose=False))
                stored = runs[0]["result"]
                diff, mism = max_abs_diff(_strip_seconds(stored), _strip_seconds(res))
                r["reproduction"] = {"status": "reproduced" if (mism == 0 and diff <= 1e-9) else "differs",
                                     "max_abs_diff": diff, "structural_mismatches": mism}
        out["records"][p.name] = r
    out["seconds"] = round(time.time() - t0, 1)
    from .. import jsonio
    jsonio.dump(jsonable(out), config.RESULTS_DIR / "evaluation.json")
    return out


def _strip_seconds(o):
    """A result without its timings (seconds differ between runs of the same code)."""
    if isinstance(o, dict):
        return {k: _strip_seconds(v) for k, v in o.items() if k not in ("seconds", "development_run")}
    if isinstance(o, list):
        return [_strip_seconds(v) for v in o]
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pre-registered runs T-A / T-C / T-S (SIM_DESIGN §4.2, §4.6, §5.8)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--date")
    f.add_argument("--only", nargs="*", choices=list(RECORD_IDS))
    f.add_argument("--tag", default="")
    f.add_argument("--supersedes")
    r = sub.add_parser("run")
    r.add_argument("rid", choices=["TA", "TC"])
    r.add_argument("--date")
    r.add_argument("--tag", default="")
    s = sub.add_parser("status")
    s.add_argument("rid", choices=["TS"])
    s.add_argument("--date")
    s.add_argument("--tag", default="")
    nt = sub.add_parser("note")
    nt.add_argument("rid", choices=list(RECORD_IDS))
    nt.add_argument("--date")
    nt.add_argument("--tag", default="")
    nt.add_argument("--text", required=True)
    nt.add_argument("--json", help="a JSON file with structured fields for the note")
    c = sub.add_parser("check")
    c.add_argument("--inputs", action="store_true")
    fc = sub.add_parser("frozen-copy")
    fc.add_argument("record")
    fc.add_argument("out")
    sub.add_parser("smoke")
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        paths = freeze(_date(a), only=a.only, tag=a.tag, supersedes=a.supersedes)
        for rid, p in paths.items():
            rec = load_record(p)
            print(f"{rid}: {rel(p)} frozen_sha256 {rec['frozen_sha256']} file_sha256 {sha256_file(p)}")
        return 0
    if a.cmd in ("run", "status", "note"):
        path = record_path(_date(a), a.rid, tag=a.tag)
        if a.cmd == "run":
            entry = run_record(path)
        else:
            rec = load_record(path)
            probs = check_record(rec)
            if probs:
                raise SystemExit(f"记录不完整：{probs}")
            started = now()
            if a.cmd == "status":
                check = {"matches": None, "note": "a status entry runs nothing; inputs are not required to match",
                         "differ_now": len(verify_inputs(rec["frozen"]["inputs"]))}
                entry = append_result(path, "status", ts_status(rec["frozen"]["protocol"]), started, check)
            else:
                body = {"text": a.text}
                if a.json:
                    body.update(json.loads(Path(a.json).read_text(encoding="utf-8")))
                check = {"matches": None, "note": "a note runs nothing and changes no earlier entry",
                         "differ_now": len(verify_inputs(rec["frozen"]["inputs"]))}
                entry = append_result(path, "note", body, started, check)
        print(f"{a.rid}: result {entry['n']} ({entry['kind']}) appended, result_sha256 {entry['result_sha256']}")
        return 0
    if a.cmd == "check":
        out = check_all(inputs=a.inputs)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 1 if any(v["problems"] for v in out.values()) else 0
    if a.cmd == "frozen-copy":
        Path(a.out).write_text(frozen_copy(load_record(Path(a.record))), encoding="utf-8")
        return 0
    if a.cmd == "smoke":
        print(json.dumps(smoke(), ensure_ascii=False, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
