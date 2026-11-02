"""Build the model card scenariolab/qmodel/card.json from scenariolab/qmodel/schema.json plus SIM_DESIGN's overrides.

    python tools/build_model_card.py            # writes card.json
    python tools/build_model_card.py --check    # exit 1 if card.json differs from a fresh build

The card is SIM_DESIGN Appendix A (generated there by design/d03_model_card.py from schema-revise-2 3.0): this script
applies the same rules and overrides, so the card and the schema cannot disagree, and adds the machine fields the
translator, the QTable validator, the UI and the quarterly panel read (group, endo / timing codes, permission code,
L0 and financial-conditions roles, panel columns, keywords of the stated-only fields).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenariolab.qmodel import card as qcard  # noqa: E402
from scenariolab.qmodel import config_q as cq  # noqa: E402

# ---------------------------------------------------------------- design decisions (SIM_DESIGN §4.1, §5.1, §7.1)
L0_CORE = set(cq.L0_FIELDS)          # the factor round's F (tp10 only as tp10 x 1{core > 4})
SCOPED_L0 = {"d_metals_oil": "能源"}
FC_REQUIRED = {"d_gz", "d_spread"}    # financial-conditions stance: at least one (or the qualitative stance)
FC_OPTIONAL = {"d_y10", "d_usd", "d_tp10", "d_y2", "d_cpbill", "d_bdlev", "d_usd_em", "d_bund", "d_tips10"}
FACTOR_CORE = {"d_cpi_goods", "d_cpi_serv", "d_ip", "d_oil", "d_ff", "d_y10", "d_spread", "d_gz", "d_usd", "d_unrate"}
OVERRIDE = {
    "y_mult": ("输出（掩码）", "Δlog CAPE 倍数：CRSP 季末价格 /（CPI × 标普 10 年平均实际 EPS），valuation 组 us_equity_lm_cape；"
               "取代 schema 的 Δlog P/Ê（减记扭曲：2007Q3–2009Q2 P/Ê +1.005 而价格 −0.468）"),
    "y_eps": ("输出（只展示）", "滚动 EPS 的变化：含减记，只展示，不作恒等式的腿；恒等式的盈利腿改为 y_fund = Δlog CPI + Δlog E10 实际"),
    "s_pe": ("s_q（只展示）", "滚动 P/E 被 2001–02、2008–09 的减记扭曲：只展示，不进模型"),
    "s_cape": ("s_q", "改用 Shiller 2026 版工作簿（E、D 到 2026-06）；MAIN 拼接版 2024–26 高 4–12.5%，要修"),
    "s_fcfy": ("s_q（只展示）", "NIPA/Z.1 全部非金融公司的代理，不是 SPY/QQQ 的自由现金流收益率；水平状态样本外无增益"),
    "x_oil_cause": ("x_q", "枚举 supply / demand / mixed / null（不再用 0/1：翻译器与 qdata 的约定相反）"),
    "d_sent": ("x_q（只在文字明说时）", "E2：同季资产收益解释其变化 adj R² 0.24；不许翻译器自选幅度"),
    "s_rstar": ("s_q（锚）", "回测：训练期末可知的规则（过去 40 季实际政策利率均值）；HLW 当前版本只展示"),
    "d_gz": ("x_q", "因子轮以 1995–2004 为验证期选入（该十年不是干净的检验）；同季资产收益解释其变化 adj R² 0.40：它是市场价格，"
             "写它接近写股权风险溢价"),
    "d_tp10": ("x_q", "ACM 用全样本参数估计（轻度前视）；在 F 里只以 TP×1{核心>4} 进入"),
    "d_spread": ("x_q", "1962–1985 的计时已修正（因子轮）；同季资产收益解释其变化 adj R² 0.32"),
    "d_bs": ("x_q（政策，经冲击 u_bs）", "QE 对资产学不出（26 季、4 个计划）：资产效应只经由 d_tp10、d_y2、d_gz 的干预路径"),
}
ZH = {"y_mult": "倍数变化：Δlog CAPE 倍数（美股，CRSP 价格口径）", "y_eps": "滚动 EPS 变化（含减记，只展示）"}
TIMING = {"start": "期初（前定）", "real": "同季·实现", "expect": "同季·市场/调查预期", "lagq": "滞后一季",
          "path": "情景路径（假设）", "anticip": "预告路径"}
ENDO = {"E0": "E0 外生宏观", "E0p": "E0p 政策（系统+冲击）", "E1c": "E1c 商品价格", "E1": "E1 金融市场", "E2": "E2 调查/新闻",
        "E3": "E3 用股价构造", "S": "状态", "C": "情景参数", "V": "视图", "O": "输出"}
# the words a quote must contain for an E1 / E1c / E2 entry (SIM_DESIGN §7.3; the validator's keyword check)
KEYWORDS = {
    "d_bdlev": ["券商杠杆", "交易商杠杆", "非银杠杆", "去杠杆", "杠杆"],
    "d_bund": ["德国国债", "德债", "欧洲利率", "全球利率", "海外收益率", "外国国债"],
    "d_cpbill": ["商业票据", "融资市场", "短期融资", "货币市场", "流动性"],
    "d_hpi": ["房价", "住房价格", "楼市"],
    "d_usd_em": ["新兴市场货币", "新兴市场汇率", "新兴市场", "本币贬值"],
    "d_cre": ["商业地产", "写字楼", "商业不动产"],
    "d_gz": ["信用利差", "公司债利差", "GZ", "高收益", "信用"],
    "d_spread": ["信用利差", "公司债利差", "Baa", "评级利差", "信用"],
    "d_tips10": ["实际利率", "实际收益率", "TIPS", "通胀保值"],
    "d_tp10": ["期限溢价", "长端溢价", "债券义警", "长债"],
    "d_usd": ["美元", "汇率"],
    "d_y10": ["10 年期", "十年期", "长期利率", "长端", "国债收益率", "长债收益率"],
    "d_y2": ["2 年期", "两年期", "短端", "加息预期", "降息预期", "利率预期"],
    "d_igrea": ["全球需求", "全球经济", "外需", "全球活动", "航运"],
    "d_metals_oil": ["金属", "铜", "工业金属"],
    "d_natgas": ["天然气", "气价"],
    "d_oil": ["油价", "原油", "石油"],
    "d_epu": ["政策不确定性", "不确定性"],
    "d_epu_health": ["医疗政策", "医保", "药价", "医疗改革"],
    "d_gpr": ["地缘政治", "地缘", "战争", "冲突", "恐袭"],
    "d_infexp1": ["通胀预期", "预期通胀"],
    "d_infexp_lr": ["长期通胀预期", "通胀预期", "去锚", "脱锚", "预期锚"],
    "d_sent": ["消费者信心", "信心", "消费者情绪", "情绪"],
    "x_lending_std": ["放贷标准", "信贷标准", "惜贷", "信贷收紧", "银行放贷"],
}
SHOCK_ROWS = [("u_ff", "政策利率冲击", "pp/季", "1973Q1", "d_ff"), ("u_bs", "资产负债表冲击", "pp GDP/季", "2008Q3", "d_bs"),
              ("u_def", "赤字冲击", "pp GDP/季", "1973Q1", "d_deficit"), ("u_tr", "转移支付冲击", "pp/季", "1973Q1",
                                                                          "d_transfers"),
              ("u_ptax", "个人税冲击", "pp/季", "1973Q1", "d_perstax")]
OUTPUT_EXTRA = [
    {"id": "y_fund", "zh": "美股基本面腿（CPI × 10 年平均实际 EPS）", "slot": "输出", "lead": "同期", "timing": "同季·实现",
     "endo": "输出", "route": "恒等式：y_us_equity = y_mult + y_fund", "translator": "不可写", "factor_core": "", "fc": "",
     "unit": "log", "first": "1973Q1", "lag": "E 滞后约 3 个月", "note": "平滑：主要是 CPI 传导（valuation：R² 0.56，系数 0.97）"},
    {"id": "y_pd", "zh": "美股 P/D 变化（收入腿用）", "slot": "输出", "lead": "同期", "timing": "同季·实现", "endo": "输出",
     "route": "恒等式：收入 = D/P", "translator": "不可写", "factor_core": "", "fc": "", "unit": "log", "first": "1973Q1",
     "lag": "0", "note": "约 30% 的 1973–2025 P/D 上升是派息率下降，不当作估值"},
]
PANEL_GROUPS = {"y_assets": [f"y_{a}" for a in cq.ASSETS],
                "s_w": ["s_w_energy", "s_w_industrials", "s_w_staples", "s_w_healthcare", "s_w_utilities",
                        "s_w_financials", "s_w_buseq", "s_w_other"]}


def slot(f: dict) -> str:
    s = f["slot"]
    if f["layer"] == "OUT":
        return "输出"
    if f["layer"] == "A":
        return "c_q 锚"
    if f["layer"] == "view" or f["endo"] == "V":
        return "视图"
    if s.startswith("s_q") or f["layer"] in ("L0", "S"):
        return "s_q"
    return "x_q"


def asset_route(fid: str, f: dict) -> str:
    if fid in L0_CORE:
        return "L0 资产方程（因子轮 F）" + ("：只作 TP×1{核心>4}" if fid == "d_tp10" else "")
    if fid in SCOPED_L0:
        return f"L0 专用（{SCOPED_L0[fid]}）"
    t = f["tier"]
    if f["layer"] == "OUT":
        return "—"
    if t in ("1", "1s") and slot(f) == "x_q":
        return "候选（T-A 预注册检验决定）；之前只经由因子层"
    if t == "2" and slot(f) == "x_q":
        return "只经由因子层"
    if slot(f) == "s_q":
        return {"learned": "因子层的条件状态", "identity": "恒等式", "anchor": "锚"}.get(f["entry"], f["entry"])
    if slot(f) == "c_q 锚":
        return "锚（情景层给，默认可知值）"
    return "不进模型"


def permission(f: dict) -> str:
    if f["layer"] == "OUT":
        return "never"
    if slot(f) == "视图":
        return "read_only"
    if slot(f) == "s_q":
        return "data"
    if slot(f) == "c_q 锚":
        return "default"
    if f["endo"] in ("E1", "E1c", "E2"):
        return "stated_only"
    if f["endo"] == "E0p":
        return "raw_or_shock"
    return "free"


def translator(f: dict) -> str:
    if f["layer"] == "OUT":
        return "不可写（评估时掩码；情景估值只能写 c_q.valuation 的定性模式）"
    return qcard.PERMISSIONS[permission(f)]


def lead_tag(f: dict) -> str:
    t = f["timing"]
    if t in ("real", "expect"):
        return "同期"
    if t == "lagq":
        return "同期（描述上一季，季中发布）"
    if t == "start":
        return "前定（期初已知）"
    return "假设（情景给）"


def _group(sl: str, fid: str) -> str:
    return {"x_q": "x", "s_q": "s", "c_q 锚": "c", "输出": "y", "视图": "v"}[sl]


def build_card(schema: dict) -> dict:
    """The card from the project schema (SIM_DESIGN Appendix A rules and overrides + machine fields)."""
    rows = []
    order = {"x_q": 0, "s_q": 1, "c_q 锚": 2, "输出": 3, "视图": 4}
    for f in schema["fields"]:
        fid = f["id"]
        sl = slot(f)
        ov = OVERRIDE.get(fid)
        fc = "金融条件（必填其一）" if fid in FC_REQUIRED else ("金融条件（可选）" if fid in FC_OPTIONAL else "")
        r = {"id": fid, "zh": ZH.get(fid, f["zh"]), "slot": ov[0] if ov else sl, "lead": lead_tag(f),
             "timing": TIMING.get(f["timing"], f["timing"]), "endo": ENDO.get(f["endo"], f["endo"]),
             "route": asset_route(fid, f), "translator": translator(f),
             "factor_core": "是" if fid in FACTOR_CORE else "", "fc": fc, "unit": f["unit"],
             "first": str(f.get("first", "")).split("（")[0], "lag": str(f.get("lag", "")),
             "note": ov[1] if ov else ""}
        r.update({"group": _group(sl, fid), "endo_code": f["endo"], "timing_code": f["timing"], "layer": f["layer"],
                  "tier": f["tier"], "family": f.get("family", ""), "l0": fid in L0_CORE,
                  "l0_scope": SCOPED_L0.get(fid), "fc_role": "required_one" if fid in FC_REQUIRED else (
                      "optional" if fid in FC_OPTIONAL else None),
                  "permission": permission(f), "definition": f.get("definition", ""),
                  "keywords": KEYWORDS.get(fid, []), "_o": order[sl], "_l": f["layer"]})
        if f.get("enum"):
            r["enum"], r["model_value"] = list(f["enum"]), dict(f.get("model_value", {}))
        rows.append(r)
    rows.sort(key=lambda r: (r["_o"], r["_l"], r["id"]))
    shocks = []
    for u, zh, unit, first, raw in SHOCK_ROWS:
        shocks.append({"id": u, "zh": zh, "slot": "x_q（干预）", "lead": "同期", "timing": "同季·实现", "endo": "E0p 冲击",
                       "route": "经由因子层（递归识别：同季只动 E1/E2，宏观滞后反应）",
                       "translator": qcard.PERMISSIONS["intervention_only"], "factor_core": "", "fc": "", "unit": unit,
                       "first": first, "lag": "—", "note": "系统部分由反应函数/稳定器在训练期末前拟合",
                       "group": "u", "endo_code": "E0p", "timing_code": "real", "layer": "shock", "tier": "",
                       "family": "政策冲击", "l0": False, "l0_scope": None, "fc_role": None,
                       "permission": "intervention_only", "raw_field": raw, "keywords": [],
                       "definition": f"{raw} 减去系统部分（反应函数或自动稳定器，拟合到训练期末）：只在文字写明干预时给"})
    extra = []
    for e in OUTPUT_EXTRA:
        extra.append({**e, "group": "y", "endo_code": "O", "timing_code": "real", "layer": "OUT", "tier": "—",
                      "family": "输出", "l0": False, "l0_scope": None, "fc_role": None, "permission": "never",
                      "keywords": [], "definition": e["route"]})
    x_rows = [r for r in rows if r["group"] == "x"]
    allrows = x_rows + shocks + [r for r in rows if r["group"] in ("s", "c")] + \
        [r for r in rows if r["group"] == "y"] + extra + [r for r in rows if r["group"] == "v"]
    for r in allrows:
        r.pop("_o", None)
        r.pop("_l", None)
        r["panel"] = PANEL_GROUPS.get(r["id"], [r["id"]]) if r["group"] in ("x", "s", "y", "v") else []
    return {"card": cq.CARD_ID, "schema_version": schema["version"], "generated_by": "tools/build_model_card.py",
            "columns": [{"key": k, "zh": z} for k, z in qcard.COLUMNS],
            "groups": qcard.GROUP_NAMES, "permissions": qcard.PERMISSIONS, "rows": allrows}


def render(card: dict) -> str:
    return json.dumps(card, ensure_ascii=False, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--schema", type=Path, default=cq.SCHEMA_PATH)
    ap.add_argument("--out", type=Path, default=cq.CARD_PATH)
    ap.add_argument("--check", action="store_true", help="compare with the existing card instead of writing it")
    args = ap.parse_args(argv)
    card = build_card(qcard.load_schema(args.schema))
    problems = qcard.validate_card(card)
    if problems:
        print("[build_model_card] invalid card:\n  " + "\n  ".join(problems))
        return 1
    text = render(card)
    if args.check:
        same = args.out.exists() and args.out.read_text(encoding="utf-8") == text
        print(f"[build_model_card] {args.out} {'is up to date' if same else 'differs from a fresh build'}")
        return 0 if same else 1
    args.out.write_text(text, encoding="utf-8")
    counts = {g: sum(r['group'] == g for r in card['rows']) for g in qcard.GROUP_COUNTS}
    print(f"[build_model_card] {len(card['rows'])} rows {counts} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
