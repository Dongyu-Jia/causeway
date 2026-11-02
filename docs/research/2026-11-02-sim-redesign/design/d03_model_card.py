"""d03 -- the model card of SIM_DESIGN Appendix A, generated from schema.json (schema-revise-2 3.0) plus the design's
overrides (critique fixes and adoption decisions), so the two cannot disagree.
Read-only on everything except this folder's work/. Run: <venv python> -I -B d03_model_card.py <schema.json> <outdir>
"""
import csv, json, sys
from pathlib import Path

schema = json.load(open(sys.argv[1]))
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
F = {f["id"]: f for f in schema["fields"]}

# ---- design decisions --------------------------------------------------------------------------------------------
L0_CORE = {"d_oil", "d_ip", "d_cpi_goods", "d_cpi_serv", "d_ff", "d_y10", "d_spread", "d_usd", "d_unrate", "d_gz",
           "d_tp10"}                       # factor round F (tp10 only as tp10 x 1{core>4}); metals_oil for energy only
SCOPED_L0 = {"d_metals_oil": "能源"}
FC_REQUIRED = {"d_gz", "d_spread"}         # financial-conditions stance: at least one stated (or qualitative stance)
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
    "d_gz": ("x_q", "因子轮以 1995–2004 为验证期选入（该十年不是干净的检验）；同季资产收益解释其变化 adj R² 0.40：它是市场价格，写它接近写股权风险溢价"),
    "d_tp10": ("x_q", "ACM 用全样本参数估计（轻度前视）；在 F 里只以 TP×1{核心>4} 进入"),
    "d_spread": ("x_q", "1962–1985 的计时已修正（因子轮）；同季资产收益解释其变化 adj R² 0.32"),
    "d_bs": ("x_q（政策，经冲击 u_bs）", "QE 对资产学不出（26 季、4 个计划）：资产效应只经由 d_tp10、d_y2、d_gz 的干预路径"),
}
ZH = {"y_mult": "倍数变化：Δlog CAPE 倍数（美股，CRSP 价格口径）", "y_eps": "滚动 EPS 变化（含减记，只展示）"}
TIMING = {"start": "期初（前定）", "real": "同季·实现", "expect": "同季·市场/调查预期", "lagq": "滞后一季",
          "path": "情景路径（假设）", "anticip": "预告路径"}
ENDO = {"E0": "E0 外生宏观", "E0p": "E0p 政策（系统+冲击）", "E1c": "E1c 商品价格", "E1": "E1 金融市场", "E2": "E2 调查/新闻",
        "E3": "E3 用股价构造", "S": "状态", "C": "情景参数", "V": "视图", "O": "输出"}

def slot(f):
    s = f["slot"]
    if f["layer"] == "OUT": return "输出"
    if f["layer"] == "A": return "c_q 锚"
    if f["layer"] == "view" or f["endo"] == "V": return "视图"
    if s.startswith("s_q") or f["layer"] in ("L0", "S"): return "s_q"
    return "x_q"

def asset_route(fid, f):
    if fid in L0_CORE: return "L0 资产方程（因子轮 F）" + ("：只作 TP×1{核心>4}" if fid == "d_tp10" else "")
    if fid in SCOPED_L0: return f"L0 专用（{SCOPED_L0[fid]}）"
    t = f["tier"]
    if f["layer"] == "OUT": return "—"
    if t in ("1", "1s") and slot(f) == "x_q": return "候选（T-A 预注册检验决定）；之前只经由因子层"
    if t == "2" and slot(f) == "x_q": return "只经由因子层"
    if slot(f) == "s_q": return {"learned": "因子层的条件状态", "identity": "恒等式", "anchor": "锚"}.get(f["entry"], f["entry"])
    if slot(f) == "c_q 锚": return "锚（情景层给，默认可知值）"
    return "不进模型"

def translator(fid, f):
    if f["layer"] == "OUT": return "不可写（评估时掩码；情景估值只能写 c_q.valuation 的定性模式）"
    if slot(f) == "视图": return "只读"
    if slot(f) == "s_q": return "不可写（数据）；反事实起点只在实验室"
    if slot(f) == "c_q 锚": return "可写（有默认值）"
    if f["endo"] in ("E1", "E1c", "E2"): return "只在文字明说时（数字或定性幅度）"
    if f["endo"] == "E0p": return "写原始值（因经济而动）或写冲击 u（干预）"
    return "可写"

def lead_tag(f):
    t = f["timing"]
    if t in ("real", "expect"): return "同期"
    if t == "lagq": return "同期（描述上一季，季中发布）"
    if t == "start": return "前定（期初已知）"
    return "假设（情景给）"

rows = []
order = {"x_q": 0, "s_q": 1, "c_q 锚": 2, "输出": 3, "视图": 4}
for fid, f in F.items():
    sl = slot(f)
    ov = OVERRIDE.get(fid)
    note = ov[1] if ov else ""
    fc = "金融条件（必填其一）" if fid in FC_REQUIRED else ("金融条件（可选）" if fid in FC_OPTIONAL else "")
    rows.append({"字段": fid, "中文名": ZH.get(fid, f["zh"]), "槽位": ov[0] if ov else sl, "同期/前定": lead_tag(f),
                 "时点": TIMING.get(f["timing"], f["timing"]), "内生性": ENDO.get(f["endo"], f["endo"]),
                 "到资产的路径": asset_route(fid, f), "翻译器": translator(fid, f), "核心因子层": "是" if fid in FACTOR_CORE else "",
                 "金融条件": fc, "单位": f["unit"], "起始": str(f.get("first", "")).split("（")[0],
                 "发布滞后": str(f.get("lag", ""))[:28], "注": note, "_o": order.get(sl if not (ov and ov[0].startswith("s_q")) else "s_q", 5), "_l": f["layer"]})
rows.sort(key=lambda r: (r["_o"], r["_l"], r["字段"]))
# add the policy shocks and the new identity outputs (not schema fields)
extra = [
    {"字段": u, "中文名": zh, "槽位": "x_q（干预）", "同期/前定": "同期", "时点": "同季·实现", "内生性": "E0p 冲击", "到资产的路径": "经由因子层（递归识别：同季只动 E1/E2，宏观滞后反应）",
     "翻译器": "只在文字写明“干预”时（数字或定性幅度）", "核心因子层": "", "金融条件": "", "单位": unit, "起始": first, "发布滞后": "—", "注": "系统部分由反应函数/稳定器在训练期末前拟合"}
    for u, zh, unit, first in [("u_ff", "政策利率冲击", "pp/季", "1973Q1"), ("u_bs", "资产负债表冲击", "pp GDP/季", "2008Q3"),
                               ("u_def", "赤字冲击", "pp GDP/季", "1973Q1"), ("u_tr", "转移支付冲击", "pp/季", "1973Q1"),
                               ("u_ptax", "个人税冲击", "pp/季", "1973Q1")]]
extra += [{"字段": "y_fund", "中文名": "美股基本面腿（CPI × 10 年平均实际 EPS）", "槽位": "输出", "同期/前定": "同期", "时点": "同季·实现",
           "内生性": "输出", "到资产的路径": "恒等式：y_us_equity = y_mult + y_fund", "翻译器": "不可写", "核心因子层": "", "金融条件": "",
           "单位": "log", "起始": "1973Q1", "发布滞后": "E 滞后约 3 个月", "注": "平滑：主要是 CPI 传导（valuation：R² 0.56，系数 0.97）"},
          {"字段": "y_pd", "中文名": "美股 P/D 变化（收入腿用）", "槽位": "输出", "同期/前定": "同期", "时点": "同季·实现", "内生性": "输出",
           "到资产的路径": "恒等式：收入 = D/P", "翻译器": "不可写", "核心因子层": "", "金融条件": "", "单位": "log", "起始": "1973Q1",
           "发布滞后": "0", "注": "约 30% 的 1973–2025 P/D 上升是派息率下降，不当作估值"}]
cols = ["字段", "中文名", "槽位", "同期/前定", "时点", "内生性", "到资产的路径", "翻译器", "核心因子层", "金融条件", "单位", "起始", "发布滞后", "注"]
allrows = [r for r in rows if r["槽位"].startswith("x_q")] + extra[:5] + [r for r in rows if not r["槽位"].startswith("x_q")] + extra[5:]
with open(out / "model_card.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(allrows)
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for r in allrows:
    md.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |")
(out / "model_card.md").write_text("\n".join(md) + "\n", encoding="utf-8")
from collections import Counter
print(len(allrows), Counter(r["槽位"].split("（")[0] for r in allrows))
print(Counter(r["到资产的路径"].split("（")[0] for r in allrows))
