"""Synthetic episodes for the N-synth arm (SIM_DESIGN §8.3): narratives written from the model's own simulated paths.

Truth is the simulated path, so recognition is impossible and the arm measures translation fidelity alone. At each of
four train ends (config_q.TF_ENDS: 1994Q4, 2004Q4, 2014Q4, 2026Q3; origin = the next quarter) the quarterly stack is
fitted with the train-end anchors, PATHS unconditional paths of H quarters are simulated, and PER_ORIGIN of them (a
seeded draw among the paths that stay inside history) become episodes. The writer is code, not an LLM: per quarter,
the policy rate's change, unemployment's change, IP growth, the oil price and CPI y/y, rounded as an N1 narrative
rounds them; the n2 text adds a 金融条件 sentence (the GZ and Baa spreads, the 10-year yield, the dollar). The texts
carry no year, name or place, and say 「覆盖期约 N 个季度」.

The writer also returns the reference translation (what a perfect translator writes for its own text: every value
quoted from the text, units as the card's), which the mock backend replays, and the truth (the path's quarterly
asset returns and x fields), which goes into the sealed key: the scorer reads it, the runner never does.

Nothing here calls an LLM or downloads anything.
"""
from __future__ import annotations

import math

import numpy as np

from ...qmodel import config_q as cq
from . import common as C

TRAIN_ENDS = tuple(cq.TF_ENDS)
PER_ORIGIN = 2
H = 8
PATHS = 200
SEED = 20261102
FC_FIELDS = ("d_gz", "d_spread", "d_y10", "d_usd")
TRUTH_FIELDS = ("d_ff", "d_unrate", "d_ip", "d_oil", "d_cpi_goods", "d_cpi_serv", "d_gz", "d_spread", "d_y10", "d_usd",
                "d_tp10", "d_metals_oil")
NOTE = "合成片段：叙事由代码根据季度模型自己的模拟路径写成；参考翻译就是写进叙事的数字"


def _pp(v: float, up: str, down: str, flat: str, nd: int = 2, unit: str = "个百分点", eps: float = 0.05):
    """(phrase, value or None for "flat") of a change in pp."""
    r = round(float(v), nd)
    if abs(r) < eps:
        return flat, None
    word = up if r > 0 else down
    sep = "" if unit == "%" else " "
    return f"{word}约 {abs(r):.{nd}f}{sep}{unit}", r


def _pct(v_log: float, up: str, down: str, flat: str):
    """(phrase, log% value) of a log% change written as a rounded percent change."""
    p = round(100.0 * (math.exp(float(v_log) / 100.0) - 1.0))
    if p == 0:
        return flat, 0.0
    word = up if p > 0 else down
    return f"{word}约 {abs(p)}%", 100.0 * math.log1p(p / 100.0)


def _entry(value=None, sign=None, src="stated", quote=""):
    e = {"value": float(value)} if value is not None else {"sign": sign}
    e.update(src=src, quote=quote)
    return e


def write(path: dict, start: dict, *, n_quarters: int = H) -> dict:
    """Narratives n1 / n2 and their reference QTables from one path: {"n1", "n2", "reference": {"n1", "n2"}}.

    path: {field: (H,) values} with d_ff, d_unrate, d_ip, d_oil, cpi_yoy (quarter-end level) and, for n2, d_gz,
    d_spread, d_y10, d_usd; start: the levels at the origin (s_ff, s_unrate, s_cpi_yoy)."""
    head = ["# 情景叙事", "", f"覆盖期约 {n_quarters} 个季度（下文“第 k 个季度”从下个季度起算）。", "",
            f"**起点。** 政策利率约 {start['s_ff']:.1f}%，失业率约 {start['s_unrate']:.1f}%，CPI 同比约 {start['s_cpi_yoy']:.1f}%。"]
    body1, body2 = [], []
    rows1, rows2, cons = [], [], []
    for k in range(n_quarters):
        q = k + 1
        lab = f"第 {q} 个季度"
        ff_t, ff_v = _pp(path["d_ff"][k], "上调", "下调", "维持不变")
        u_t, u_v = _pp(path["d_unrate"][k], "上升", "下降", "基本不变", nd=1)
        ip_t, ip_v = _pp(path["d_ip"][k], "增长", "下降", "基本持平", nd=1, unit="%")
        oil_t, oil_v = _pct(path["d_oil"][k], "上涨", "下跌", "基本不变")
        cpi = round(float(path["cpi_yoy"][k]), 1)
        s_ff, s_u, s_ip, s_oil, s_cpi = (f"{lab}，政策利率{ff_t}", f"{lab}，失业率{u_t}", f"{lab}，工业产出比上季{ip_t}",
                                         f"{lab}，油价{oil_t}", f"{lab}末，CPI 同比约 {cpi:.1f}%")
        body1.append(f"**{lab}。** " + "；".join([s_ff, s_u, s_ip, s_oil, s_cpi]) + "。")
        x = {"d_ff": _entry(ff_v if ff_v is not None else 0.0, src="stated", quote=s_ff),
             "d_unrate": (_entry(u_v, src="stated", quote=s_u) if u_v is not None else
                          _entry(sign="0", src="stated_qual", quote=s_u)),
             "d_ip": (_entry(ip_v, src="stated", quote=s_ip) if ip_v is not None else
                      _entry(sign="0", src="stated_qual", quote=s_ip)),
             "d_oil": _entry(oil_v, src="derived", quote=s_oil)}
        cons.append({"kind": "level", "field": "s_cpi_yoy", "quarters": [q], "value": cpi, "sd": 0.1, "quote": s_cpi})
        rows1.append({"q": q, "x": dict(x), "fc": {"stance": "未说明"}})
        gz_t, gz_v = _pp(path["d_gz"][k], "上升", "下降", "基本不变")
        sp_t, sp_v = _pp(path["d_spread"][k], "走阔", "收窄", "基本不变")
        y_t, y_v = _pp(path["d_y10"][k], "上升", "下降", "基本不变")
        usd_t, usd_v = _pct(path["d_usd"][k], "升值", "贬值", "基本不变")
        s_gz, s_sp, s_y, s_usd = (f"{lab}，GZ 信用利差{gz_t}", f"{lab}，Baa 信用利差{sp_t}", f"{lab}，10 年期国债收益率{y_t}",
                                  f"{lab}，美元{usd_t}")
        body2.append(body1[-1] + " 金融条件：" + "；".join([s_gz, s_sp, s_y, s_usd]) + "。")
        x2 = dict(x)
        for f, v, s in (("d_gz", gz_v, s_gz), ("d_spread", sp_v, s_sp), ("d_y10", y_v, s_y)):
            x2[f] = _entry(v, src="stated", quote=s) if v is not None else _entry(sign="0", src="stated_qual", quote=s)
        x2["d_usd"] = _entry(usd_v, src="derived", quote=s_usd)
        rows2.append({"q": q, "x": x2, "fc": {"stance": "未说明"}})

    def table(rows):
        return {"schema": "qtable/1", "model_card": cq.CARD_ID, "n_quarters": n_quarters, "rows": rows,
                "c_q": {"constraints": [dict(c) for c in cons]}, "events": [], "notes": [NOTE]}

    n1 = "\n".join(head + [""] + ["\n\n".join(body1)]) + "\n"
    n2 = "\n".join(head + [""] + ["\n\n".join(body2)]) + "\n"
    return {"n1": n1, "n2": n2, "reference": {"n1": table(rows1), "n2": table(rows2)}}


def episodes(Q=None, *, train_ends=TRAIN_ENDS, per_origin: int = PER_ORIGIN, n_quarters: int = H, paths: int = PATHS,
             seed: int = SEED, n_boot: int | None = None) -> list[dict]:
    """The synthetic episodes: [{"train_end", "origin", "path", "seed", "n_quarters", "texts" (write()), "truth"
    {"assets" (H, 13), "x" {field: (H,)}}, "start"}] in train-end order."""
    from ...qmodel import rollout as R
    out = []
    for i, te in enumerate(train_ends):
        stack = R.fit_stack(te, anchors="train_end", seed=seed, n_boot=n_boot, Q=Q)
        roll = R.run_path(stack, [], (), H=n_quarters, n_paths=paths, seed=seed + i, cma="historical", mode="test")
        fx, ax = roll.fx, roll.ax
        tails = R.tail_flags(fx.levels["gz"], fx.levels["unrate"], ax.total[:, :, R.IA["us_equity"]]).any(axis=1)
        ok = np.flatnonzero(~tails & np.isfinite(ax.price).all(axis=(1, 2)))
        rng = np.random.default_rng([seed, i])
        pick = sorted(int(k) for k in rng.choice(ok, size=min(per_origin, len(ok)), replace=False))
        for k in pick:
            p = {f: fx.x[f][k] for f in ("d_ff", "d_unrate", "d_ip", "d_oil") + FC_FIELDS}
            p["cpi_yoy"] = fx.levels["cpi_yoy"][k]
            start = {s: float(roll.states[s][k, 0]) for s in ("s_ff", "s_unrate", "s_cpi_yoy")}
            truth = {"assets": np.round(ax.price[k], 8).tolist(),
                     "x": {f: np.round(fx.x[f][k], 6).tolist() for f in TRUTH_FIELDS if f in fx.x}}
            out.append({"train_end": te, "origin": stack.origin, "path": k, "seed": seed + i, "n_quarters": n_quarters,
                        "texts": write(p, start, n_quarters=n_quarters), "truth": truth, "start": start,
                        "stack": stack.version})
    return out


def label() -> str:
    return C.ARMS["NS"]["zh"]
