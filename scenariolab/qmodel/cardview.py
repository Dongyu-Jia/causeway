"""The 变量卡片 tab's view of the model card (SIM_DESIGN §3.3, §10.2, Appendix A): GET /api/q/card/view.

view() -> {
  schema "card-view/1", card, schema_version, label 试运行, n_rows
  today      {quarter, data_end_month, last_data_quarter} of the quarterly panel (None without data/)
  groups     [{id, zh, n}]; filters {group, lead, timing, endo, route: [{value, label, n}]}
  legend     {lead, endo, route: {value: explanation}}
  rows       [{id, zh, group, group_zh, slot, lead, lead_kind, timing, endo, endo_code, endo_short, route, route_kind,
               route_short, translator, translator_short, permission, factor_core, fc, fc_role, unit, first, lag,
               definition, note, family, keywords,
               series {col, label, first, last_q, values [...], value, text, pct, n, pct_since} | None,
               today {value, text, quarter, pct, kind} | None,
               evidence [{kind, text, href}]}]
}

Every text a reader sees is Chinese: card ids inside the generator's notes and definitions are replaced by the
variable's Chinese name in 「」 (readable()), and a few internal names (the worktree and script names of the research
rounds) by what they are. Series come from the quarterly panel (data/processed/panel_quarterly.csv): for a quarterly
change (x_q fields, outputs) "today" is the last realized quarter; for a start-of-quarter state the panel's last row
(known at the start of the current quarter). The percentile is the share of the series' own history (from its first
quarter) at or below today's value. Evidence: the factor layer's computed acceptance statistics (qmodel.json: T-C
completion by pattern, F7 fill-in skill, F8 coverage, the policy reaction functions), the asset layer, the
valuation block and the pre-registered T-A record, each with a link to the page that shows it.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from . import card as qcard
from . import config_q as cq

SCHEMA = "card-view/1"
GROUP_ZH = {"x": "季内输入 x_q", "u": "政策冲击（干预）", "s": "季初状态 s_q", "c": "长期锚 c_q", "y": "输出",
            "v": "视图（只读）"}
GROUP_ORDER = ("x", "u", "s", "c", "y", "v")
LEAD_KIND = {"同期": "同期", "前定（期初已知）": "前定", "假设（情景给）": "假设", "同期（描述上一季，季中发布）": "同期"}
LEAD_LEGEND = {"同期": "与资产回报同一个季度实现（不领先资产）", "前定": "季初已经知道（数据发布滞后之后）",
               "假设": "情景给的长期假设（锚）"}
ENDO_ORDER = ("E0", "E0p", "E1c", "E1", "E2", "S", "C", "O", "V")
ENDO_SHORT = {"E0": "E0 外生宏观", "E0p": "E0p 政策", "E1c": "E1c 商品价格", "E1": "E1 金融市场", "E2": "E2 调查与新闻",
              "S": "状态", "C": "情景参数", "O": "输出", "V": "视图"}
ENDO_LEGEND = {"E0": "外生宏观：文字可写", "E0p": "政策：可写原始值（因经济而动）或写干预冲击",
               "E1c": "商品价格：只在文字明说时写", "E1": "金融市场价格：只在文字明说时写（写它接近写股权风险溢价）",
               "E2": "调查与新闻指数：只在文字明说时写", "S": "季初状态：来自数据", "C": "情景参数（长期锚）",
               "O": "模型输出：不可写", "V": "视图：只读，不进模型"}
ROUTES = {"L0": "L0 资产方程", "candidate": "候选（T-A）", "factor": "只经由因子层", "shock": "因子层（干预）",
          "state": "条件状态", "identity": "恒等式", "anchor": "锚", "output": "模型输出", "none": "不进模型"}
ROUTE_LEGEND = {"L0": "资产方程里有载荷（因子轮 F 的月度 β 加总到季度）",
                "candidate": "资产层的候选：只有通过预注册检验 T-A 才进入资产方程，之前只经由因子层",
                "factor": "不进资产方程：只通过与 L0 字段的联合分布影响资产",
                "shock": "政策冲击：按递归识别，同季只动金融市场与调查，宏观滞后反应",
                "state": "季初状态：给因子层作条件", "identity": "由会计恒等式滚动（估值、收入、权重）",
                "anchor": "长期锚：拉动因子层的稳态", "output": "模型的输出", "none": "只展示，不进模型"}
PERM_SHORT = {"free": "可写", "stated_only": "只在文字明说时", "raw_or_shock": "原始值或干预冲击",
              "intervention_only": "只在写明干预时", "default": "可写（有默认值）", "data": "不可写（数据）",
              "read_only": "只读", "never": "不可写"}
# the series a multi-column row shows, and the name of that column
REPRESENTATIVE = {"s_w": ("s_w_buseq", "科技（BusEq）的市值权重"), "y_assets": ("y_us_equity", "美股的季度对数价格回报")}
# log levels shown as a level (CAPE 41.2 倍)
LEVEL_OF_LOG = {"s_cape": ("CAPE", 1), "s_pd": ("P/D", 1), "s_pe": ("P/E", 1)}
ANCHOR_DEFAULT = {"a_spread": ("spread", "pp"), "a_ustar": ("unrate", "%"), "a_tp": ("term_premium", "pp"),
                  "a_oil": (None, "log"), "a_usd": (None, "log")}
OIL_CAUSE = {0.0: "需求推动", 0.5: "供给与需求混合", 1.0: "供给冲击"}
CATEGORICAL = {"x_oil_cause"}                      # a percentile of a category means nothing
F7_FIELD = {"ps": "d_cpi_serv", "ff": "d_ff", "y10": "d_y10", "sp": "d_spread", "gz": "d_gz", "ur": "d_unrate",
            "usd": "d_usd"}
TC_PATTERN = {"a": "随机缺一半字段", "b": "只给宏观与政策", "c": "只给金融条件"}
POLICY_ROWS = {"d_ff": "u_ff", "d_deficit": "u_def", "d_transfers": "u_tr", "d_perstax": "u_ptax", "d_bs": "u_bs"}
VALUATION_ROWS = {"s_cape", "s_pd", "s_pe", "s_fcfy", "s_ecyc", "s_payout", "y_mult", "y_fund", "y_pd", "y_eps"}
# ids the generator's texts use that are not card rows, and internal names of the research rounds
EXTRA_NAMES = {"s_infl_regime": "高通胀体制标志", "s_own_pd": "行业自身 P/D", "y_us_equity": "美股价格回报",
               "s_zlb": "零下限标志", "s_core4": "高核心通胀标志", "us_equity_lm_cape": "CAPE 倍数序列",
               "tp10_x_core4": "期限溢价 × 高核心通胀", "r_eff": "有效利率"}
WORDS = [(re.compile(r"高通胀体制 s_infl_regime"), "高通胀体制"), (re.compile(r"c_q\.valuation"), "情景表的估值陈述"), (re.compile(r"(?<![A-Za-z])FWT(?![A-Za-z])"), "因子轮"),
         (re.compile(r"(?<![A-Za-z])MAIN(?![A-Za-z])"), "原应用"), (re.compile(r"(?<![A-Za-z_])qdata(?![A-Za-z_])"), "研究轮的数据脚本"),
         (re.compile(r"schema_lit"), "文献版字段表"), (re.compile(r"valuation 组"), "估值研究"),
         (re.compile(r"(?<![A-Za-z_])schema(?![A-Za-z_])"), "原字段表"), (re.compile(r"(?<![A-Za-z])Given(?![A-Za-z])"), "给定值"),
         (re.compile(r"(?<![A-Za-z])oracle(?![A-Za-z])"), "已知路径检验"), (re.compile(r"TP×core4"), "期限溢价 × 高核心通胀"),
         (re.compile(r"(?<![A-Za-z_])policy_stance(?![A-Za-z_])"), "政策立场档"),
         (re.compile(r"(?<![A-Za-z_.])valuation(?![A-Za-z_])"), "估值研究")]
ID_RE = re.compile(r"(?<![A-Za-z0-9_])([a-z]{1,2}_[a-z0-9_]+)(?![A-Za-z0-9_])")
CJK = "\u4e00-\u9fff，。；：、（）"
CJK_BEFORE = re.compile(f"(?<=[{CJK}])[ \t]+(?=「)")
CJK_AFTER = re.compile(f"(?<=」)[ \t]+(?=[{CJK}])")
CJK_GAP = re.compile("(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])")      # left where a Latin name was replaced
LAGGED = {"l_d_ff": "上季政策利率变动", "l_d_gz": "上季 GZ 利差变动", "l_d_unrate": "上季失业率变动",
          "l_d_perstax": "上季个人税负担变动", "l_d_bs": "上季资产负债表变动", "eq4_lag": "过去 4 季股票回报",
          "zlb": "零下限标志", "zlb_x_dun": "零下限 × 失业率变动", "const": "常数"}
TEXT_KEYS = ("zh", "slot", "lead", "timing", "endo", "route", "translator", "fc", "unit", "first", "lag", "note",
             "definition", "family")


def readable(text: str, names: dict) -> str:
    """A card text with every card id (and the few other ids the generator wrote) as 「中文名」."""
    s = str(text or "")
    for pat, rep in WORDS:
        s = pat.sub(rep, s)

    def sub(m):
        tok = m.group(1)
        zh = names.get(tok) or EXTRA_NAMES.get(tok)
        return f"「{zh}」" if zh else tok
    s = ID_RE.sub(sub, s)
    # no space between a 「name」 and Chinese text or punctuation next to it (the ids were set off by spaces)
    return CJK_GAP.sub("", CJK_AFTER.sub("", CJK_BEFORE.sub("", s)))


def route_kind(r: dict) -> str:
    route = r.get("route") or ""
    if r.get("l0") or route.startswith("L0"):
        return "L0"
    if "候选" in route:
        return "candidate"
    if r.get("group") == "u" or "递归识别" in route:
        return "shock"
    if route.startswith("只经由因子层"):
        return "factor"
    if "条件状态" in route:
        return "state"
    if route.startswith("恒等式"):
        return "identity"
    if route.startswith("锚"):
        return "anchor"
    if r.get("group") == "y":
        return "output"
    return "none"


def _sig(x: float, d: int = 4):
    if x is None or not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{d}g}") + 0.0


def _fmt(row: dict, v: float, col: str | None = None) -> str:
    """Today's value in the row's unit, as a reader sees it (signed for quarterly changes, levels unsigned)."""
    if v is None or not math.isfinite(v):
        return "—"
    rid, unit, g = row["id"], row.get("unit") or "", row["group"]
    signed = g in ("x", "u", "y")
    if rid == "x_oil_cause":
        return OIL_CAUSE.get(round(v * 2) / 2, f"{v:.2f}")
    if rid in LEVEL_OF_LOG:
        name, d = LEVEL_OF_LOG[rid]
        return f"{name} {math.exp(v):.{d}f} 倍"
    if col == "y_us_equity" or rid in ("y_mult", "y_fund", "y_pd", "y_eps"):
        return f"{math.expm1(v) * 100:+.1f}%"
    if unit.startswith("pp"):
        return f"{v:+.2f} pp" if signed else f"{v:.2f} pp"
    if unit.startswith("%") and not signed:
        return f"{v:.2f}%"
    if unit == "%年化":
        return f"{v:+.1f}%（年化）"
    if unit.startswith("log%"):
        return f"{v:+.1f}%"
    if unit == "log":
        return f"{v:+.3f}" if signed else f"{v:.3f}"
    if unit == "%":
        return f"{v:+.2f}%"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def _load_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _panel():
    from . import panel as qpanel
    try:
        Q = qpanel.load_panel()
        info = (qpanel.load_meta() or {}).get("info") or {}
    except (OSError, ValueError, KeyError):
        return None, {}
    return Q, info


def _series(Q, col: str, label: str | None) -> dict | None:
    if Q is None or col not in Q.columns:
        return None
    s = Q[col].astype(float)
    valid = s.dropna()
    if valid.empty:
        return None
    first, last_q = str(valid.index[0]), str(valid.index[-1])
    span = s.loc[valid.index[0]:valid.index[-1]]
    last = float(valid.iloc[-1])
    return {"col": col, "label": label, "first": first, "last_q": last_q,
            "values": [_sig(float(v)) if v == v else None for v in span.tolist()],
            "value": _sig(last, 6), "pct": round(float((valid <= last).mean()), 4), "n": int(valid.size),
            "pct_since": first}


def _evidence(r: dict, names: dict, fl: dict | None, ta: dict | None) -> list[dict]:
    rid, g, kind = r["id"], r["group"], route_kind(r)
    ev = []
    if kind == "L0":
        scope = "（只进能源）" if r.get("l0_scope") else "（只以期限溢价 × 高核心通胀进入）" if rid == "d_tp10" else ""
        ev.append({"kind": "asset", "href": "#diagnostics",
                   "text": f"资产方程里有载荷{scope}：因子轮 F 的月度 β 加总到季度；月度传导的检验见“诊断与迭代”。"})
    if kind == "candidate":
        st = (ta or {}).get("status") or "missing"
        txt = {"missing": "资产层候选：预注册检验 T-A 尚无记录，现在只经由因子层影响资产。",
               "registered": f"资产层候选：T-A 已预注册（{(ta or {}).get('date')}），还没有结果；现在只经由因子层影响资产。",
               "done": "资产层候选：T-A 的结果见“情景条件检验”的预注册记录。"}.get(st, "资产层候选。")
        ev.append({"kind": "ta", "href": "#evaluation?records", "text": txt})
    if r.get("fc_role") == "required_one":
        ev.append({"kind": "fc", "href": "#evaluation?oracle",
                   "text": "金融条件（每季必填其一）：研究轮的消融里，已知因子路径时的能力几乎全部来自信用利差。"})
    if rid in VALUATION_ROWS:
        ev.append({"kind": "valuation", "href": "#valuation?quarterly",
                   "text": "估值是状态和恒等式的一条腿，不是预测量：口径、今天的分位与局限见“估值与盈利”。"})
    if fl:
        tc = ((fl.get("T_C") or {}).get("test") or {})
        bits = []
        for pat in ("b", "c", "a"):
            c = (tc.get(pat) or {}).get(rid)
            if isinstance(c, dict) and c.get("r2_vs_uncond") is not None:
                bits.append(f"{TC_PATTERN[pat]}时补全 R² {c['r2_vs_uncond']:+.2f}，80% 覆盖 {c.get('cov80', 0) * 100:.0f}%（n {c.get('n')}）")
        if bits:
            ev.append({"kind": "tc", "href": "#evaluation?engine",
                       "text": "补全检验 T-C（计算值，2015–2026 年检验期，相对无条件抽样）：" + "；".join(bits) + "。"})
        tf = fl.get("T_F") or {}
        f7k = next((k for k, f in F7_FIELD.items() if f == rid), None)
        if f7k:
            parts = []
            for h, blk in (tf.get("F7") or {}).items():
                v = ((blk or {}).get("r2") or {}).get(f7k)
                if v is None:
                    continue
                eng = ((blk or {}).get("engine") or {}).get(f7k)
                lab = f"{h} 个季度 R² {v:+.2f}" + (f"（月度引擎 {eng:+.2f}）" if eng is not None else "")
                if h in ("12", "20"):
                    lab += "（每个十年只有 2–3 个独立窗口，只能当个案）"
                parts.append(lab)
            if parts:
                ev.append({"kind": "f7", "href": "#evaluation?engine",
                           "text": "已知 CPI、工业产出和油价时的补全技能（T-F F7，1990–2026 年滚动）：" + "；".join(parts) + "。"})
        cov = ((tf.get("F8") or {}).get("per_field") or {}).get(rid)
        if cov is not None:
            ev.append({"kind": "f8", "href": "#evaluation?engine",
                       "text": f"4 个季度变化的 80% 覆盖率（T-F F8，1985 年起不重叠窗口）：{cov * 100:.0f}%（目标 72%–88%）。"})
        pol = ((fl.get("policy") or {}).get(cq.TF_ENDS[-1]) or {})
        shock = POLICY_ROWS.get(rid) or (rid if g == "u" else None)
        p = pol.get(shock) if shock else None
        if isinstance(p, dict) and p.get("r2") is not None:
            regs = "、".join(names.get(x) or LAGGED.get(x) or "其他" for x in (p.get("regressors") or []))
            samp = "只用非零下限季度" if p.get("sample") == "non_zlb" else "全部季度"
            ev.append({"kind": "policy", "href": "#evaluation?engine",
                       "text": f"系统部分（反应函数或自动稳定器，{samp}，拟合到 {p.get('train_end')}）：R² {p['r2']:.2f}，n {p.get('n')}，"
                               f"残差标准差 {p.get('resid_sd', 0):.2f}；解释变量：{regs}。冲击 = 实际值减系统部分，只在文字写明干预时给。"})
    if g == "s" and rid in ("s_core_yoy",):
        ev.append({"kind": "regime", "href": "#evaluation?engine",
                   "text": "核心通胀 > 4% 定义高通胀体制：期限溢价的交互项和股债相关的分体制检验（T-F F5）用它。"})
    return ev


@lru_cache(maxsize=4)
def _cached(key: tuple) -> dict:
    return _build()


def _stamp(path: Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def view() -> dict:
    """The card view (cached until card.json, the panel, qmodel.json or a pre-registration record changes)."""
    from ..evaluation import evalview as EV
    key = (_stamp(cq.CARD_PATH), _stamp(cq.PANEL_Q_PATH), _stamp(cq.QMODEL_RESULTS_PATH),
           tuple(sorted((str(p), _stamp(p)) for p in EV._record_files())))
    return _cached(key)


def _build() -> dict:
    card = qcard.load_card()
    rows_in = qcard.rows(card)
    names = {r["id"]: readable(r["zh"], {}) for r in rows_in}
    Q, info = _panel()
    q = _load_json(cq.QMODEL_RESULTS_PATH)
    fl = (q or {}).get("factor_layer") if isinstance(q, dict) else None
    try:
        from ..evaluation import evalview as EV
        ta = EV.tests_status(EV.records()).get("TA")
    except Exception:  # noqa: BLE001 - the records are optional here
        ta = None
    today_q = str(Q.index[-1]) if Q is not None else None
    out_rows = []
    for r in rows_in:
        g = r["group"]
        row = {k: readable(r.get(k), names) if k not in ("first",) else r.get(k) for k in TEXT_KEYS}
        kind = route_kind(r)
        row.update({"id": r["id"], "group": g, "group_zh": GROUP_ZH.get(g, g),
                    "lead_kind": LEAD_KIND.get(r.get("lead"), r.get("lead")), "endo_code": r.get("endo_code"),
                    "endo_short": ENDO_SHORT.get(r.get("endo_code"), r.get("endo")),
                    "timing_code": r.get("timing_code"), "route_kind": kind, "route_short": ROUTES[kind],
                    "translator_short": PERM_SHORT.get(r.get("permission"), r.get("translator")),
                    "permission": r.get("permission"), "factor_core": r.get("factor_core") == "是",
                    "fc_role": r.get("fc_role"), "keywords": list(r.get("keywords") or []),
                    "enum": list(r.get("enum") or []) or None})
        col, label = (REPRESENTATIVE.get(r["id"]) or ((r.get("panel") or [None])[0], None))
        ser = _series(Q, col, label) if col else None
        today = None
        if ser:
            v = ser["value"]
            state = g in ("s", "v")
            today = {"value": v, "text": _fmt(r, v, col), "quarter": ser["last_q"],
                     "pct": None if r["id"] in CATEGORICAL else ser["pct"],
                     "kind": "start" if state else "realized",
                     "when": (f"{ser['last_q']} 季初" if state else f"{ser['last_q']} 季")}
            if state and today_q and ser["last_q"] != today_q:
                today["when"] = f"{ser['last_q']} 季初（数据发布滞后）"
        elif g == "c" and r["id"] in ANCHOR_DEFAULT:
            k, unit = ANCHOR_DEFAULT[r["id"]]
            v = cq.FORWARD_ANCHORS.get(k) if k else 0.0
            today = {"value": v, "text": (f"{v:.2f}{'%' if unit == '%' else ' pp'}" if k else "0（训练期的中位数或均值）"),
                     "quarter": None, "pct": None, "kind": "default", "when": "前瞻默认（判断值）"}
        row["series"] = ser
        row["today"] = today
        row["evidence"] = _evidence(r, names, fl, ta)
        out_rows.append(row)
    out_rows.sort(key=lambda x: GROUP_ORDER.index(x["group"]) if x["group"] in GROUP_ORDER else 9)

    def counts(key, order=None, labels=None):
        seen = {}
        for x in out_rows:
            seen[x[key]] = seen.get(x[key], 0) + 1
        keys = [k for k in (order or []) if k in seen] + [k for k in seen if k not in (order or [])]
        return [{"value": k, "label": (labels or {}).get(k, k), "n": seen[k]} for k in keys]

    endo_labels = {}
    for x in out_rows:
        endo_labels.setdefault(x["endo_code"], x["endo"])
    return {
        "schema": SCHEMA, "card": card.get("card"), "schema_version": card.get("schema_version"),
        "label": cq.TRIAL_LABEL, "n_rows": len(out_rows),
        "today": ({"quarter": today_q, "data_end_month": info.get("data_end_month"),
                   "last_data_quarter": info.get("last_data_quarter")} if Q is not None else None),
        "groups": [{"id": gid, "zh": GROUP_ZH[gid], "n": sum(1 for x in out_rows if x["group"] == gid)} for gid in GROUP_ORDER],
        "filters": {"group": counts("group", list(GROUP_ORDER), GROUP_ZH),
                    "lead": counts("lead_kind", ["同期", "前定", "假设"]),
                    "timing": counts("timing"),
                    "endo": counts("endo_code", list(ENDO_ORDER), endo_labels),
                    "route": counts("route_kind", list(ROUTES), ROUTES)},
        "legend": {"lead": LEAD_LEGEND, "endo": {k: ENDO_LEGEND[k] for k in ENDO_ORDER}, "route": ROUTE_LEGEND},
        "rows": out_rows,
        "notes": ["“同期”的输入与资产回报在同一个季度实现：模型不说它们领先资产；“前定”的状态在季初已经知道。",
                  "分位：这个序列自己的历史里，不高于今天这个值的季度所占的比例（季度面板 1973 年起）。",
                  "数据是今天的修订版，不是当时的实时版本；发布滞后列给的是数据在季内什么时候可以知道。"],
    }
