"""P8b (SIM_DESIGN §9, §10.3-§10.6, §14 P8): the forecast tab 未来一年：情景条件模拟, the quarterly scenario detail, the
QTable lab and the texts. Static checks of the web modules (routing, deep links, no LLM text through innerHTML, the
wording rules), the scenario mixture's math and forward ledger, the QTable display view, the valuation view and the
endpoints. Data-dependent checks skip cleanly without data/; nothing here calls an LLM or downloads anything."""
import json
import math
import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scenariolab import config
from scenariolab.analysis import scenario_mix as SM
from scenariolab.qmodel import qview
from scenariolab.server import app

client = TestClient(app, raise_server_exceptions=False)
WEB = config.ROOT / "web" / "js"
TABS = WEB / "tabs"
OWNED = ["forecast.js", "_qdetail.js", "_qlab.js", "lab.js", "_scenario_detail.js", "method.js", "overview.js",
         "valuation.js", "backtest.js", "diagnostics.js"]
RESULT_TABS = ["forecast.js", "_qdetail.js", "_qlab.js", "_scenario_detail.js"]     # scenario results are shown here
PANEL = config.ROOT / "data" / "processed" / "panel_quarterly.csv"
needs_panel = pytest.mark.skipif(not PANEL.exists(), reason="quarterly panel not built")
needs_paths = pytest.mark.skipif(not (config.RESULTS_DIR / "scenario_paths.npz").exists() or not PANEL.exists(),
                                 reason="scenario paths or the quarterly panel not built")


def src(name: str) -> str:
    return (TABS / name).read_text(encoding="utf-8")


# ================================================================ routing and deep links

def test_the_forecast_tab_is_renamed_and_keeps_its_id():
    """§9: 一年期预测 becomes 未来一年：情景条件模拟; the id stays `forecast`, so #forecast links still work, and the other
    tabs keep their ids and order."""
    app_js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "{ id: 'forecast', name: '未来一年：情景条件模拟' }" in app_js
    assert "一年期预测" not in app_js
    ids = re.findall(r"\{ id: '([a-z_]+)', name: '", app_js)
    for a, b in [("overview", "eras"), ("regimes", "causality"), ("forecast", "backtest"), ("backtest", "diagnostics"),
                 ("diagnostics", "valuation"), ("scenarios", "lab"), ("lab", "instruments"), ("portfolio", "withdrawal")]:
        assert ids.index(a) < ids.index(b), (a, b)
    for name in ("forecast.js", "_qdetail.js", "_qlab.js", "lab.js"):
        r = client.get(f"/js/tabs/{name}")
        assert r.status_code == 200 and r.headers.get("cache-control") == "no-cache", name


def test_deep_links_are_unchanged():
    """#lab?<custom id> and #lab?tpl=<id> still open the monthly spec editor; #lab?q=<id> is the new QTable link;
    #scenarios?<id>, #valuation?<section> (one more section) and #backtest?... keep their parsing."""
    lab = src("lab.js")
    assert "if (param && !/^q=/.test(param)) mode = 'monthly';" in lab
    assert "const tpl = param.startsWith('tpl=') ? param.slice(4) : builtinIds.has(param) ? param : null;" in lab
    assert "openSaved(param, { intoEditor: true, scroll: false })" in lab
    qlab = src("_qlab.js")
    assert "const m = /^q=([a-z0-9_]{1,60})$/.exec(param || '');" in qlab
    assert "ui.setRoute('scenarios', encodeURIComponent(id), alive);" in src("scenarios.js")
    val = src("valuation.js")
    for key in ("answer", "today", "history", "regression", "kappa", "variants", "factors", "scenarios", "quarterly"):
        assert f"['{key}', " in val, key
    assert "return param && sections[param] ? sections[param] : null;" in val
    assert "const CUSTOM_RE = /^custom=" in src("backtest.js")
    # the scenario detail links the built-in's QTable into the lab with the new link, the template with the old one
    det = src("_scenario_detail.js")
    assert "`#lab?tpl=${d.id}`" in det and "`#lab?q=${d.id}`" in det


# ================================================================ no LLM text reaches innerHTML

def test_no_text_reaches_inner_html():
    """Every module sets text with h() / textContent: innerHTML is only ever cleared, no module uses ui.h's `html`
    attribute, insertAdjacentHTML, outerHTML or document.write."""
    for p in sorted(WEB.rglob("*.js")):
        if "vendor" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"innerHTML\s*=\s*([^;\n]+)", text):
            if p.name == "ui.js" and m.group(1).strip() == "v":
                continue                                           # the `html` attribute itself (unused)
            assert m.group(1).strip() in ("''", '""'), (p.name, m.group(0))
        assert "insertAdjacentHTML" not in text and "document.write" not in text, p.name
        assert not re.search(r"outerHTML\s*=", text), p.name
        if p.name != "ui.js":
            assert not re.search(r"[{,]\s*html\s*:", text), p.name


def test_chart_tooltips_escape_every_text():
    """ECharts tooltips are HTML strings: the new modules escape names, labels and values there."""
    qd = src("_qdetail.js")
    fan = qd.split("export function quarterFan")[1].split("\n}\n")[0]
    assert "ui.esc(x[i])" in fan and "ui.esc(name)" in fan and "ui.esc(refName)" in fan
    assert "const t = (v) => ui.esc(" in fan
    fc = src("forecast.js")
    sens = fc.split("const drawSens = () => {")[1].split("\n  };\n")[0]
    assert "tip: (r, c, v) => `<b>${ui.esc(nameOf(ids[r]))}</b> × ${ui.esc(assetName(cols[c]))}" in sens
    # LLM output in the lab is shown as text (a <pre> with textContent), quotes and evidence through h()
    qlab = src("_qlab.js")
    assert "h('pre', { class: 'code ql-wrap' }, JSON.stringify(res.raw, null, 2))" in qlab


# ================================================================ wording rules (§10.6)

TITLE_RE = re.compile(r"""(?:title:\s*|section-title(?:[^']*)?' \},\s*|section\()(['`])((?:(?!\1).)*)\1""")


def titles(text: str) -> list[str]:
    return [m.group(2) for m in TITLE_RE.finditer(text)]


def test_no_prediction_in_scenario_result_titles():
    """「预测」 appears only for Layer 1 (probabilities) and the ledger: no card or section title of a scenario result
    says 预测 (titles about the probabilities or the ledger may)."""
    seen = 0
    for name in RESULT_TABS:
        for t in titles(src(name)):
            seen += 1
            if "预测" in t:
                assert "概率" in t or "账本" in t, (name, t)
    assert seen > 40
    fc = src("forecast.js")
    assert "这不是预测：下面是 ${n} 个情景在未来 ${nQ} 个季度的模拟，按情景概率加权。预测只在情景概率里。" in fc
    # the replay section keeps the analog method's record, under its own name
    assert "'历史回放（相似起点法）'" in fc and "'相似起点法的滚动一年期成绩单（回放）'" in fc


def test_valuation_assumptions_always_say_they_assume_the_price():
    """估值假设 always carries 「相当于直接假设股价」: in every module line that writes it (or the two after), and in the
    QTable view's words for a valuation statement."""
    for name in OWNED:
        lines = src(name).splitlines()
        for i, line in enumerate(lines):
            if "估值假设" in line and not line.lstrip().startswith("//"):
                window = " ".join(lines[i:i + 3])
                ok = "相当于直接假设股价" in window or "VAL_WORDING" in window
                assert ok, (name, i + 1, line.strip())
    qt = {"schema": "qtable/1", "n_quarters": 2, "rows": [{"q": 1, "x": {}, "fc": {"stance": "未说明"}},
                                                         {"q": 2, "x": {}, "fc": {"stance": "未说明"}}],
          "c_q": {"valuation": [{"quarters": [1, 2], "mode": "rerate", "direction": "-", "size": "明显",
                                 "quote": "估值压缩"}]}}
    items = [it for r in qview.cells(qt) for it in r["cells"]["valuation"]["items"]]
    assert items and all("相当于直接假设股价" in it["text"] for it in items)
    # the attribution's 估值假设 part (a legend label) is explained with the wording in its card
    attr = src("_qdetail.js").split("function attributionBlock")[1].split("\n}\n")[0]
    assert "其中“估值假设”${VAL_WORDING}" in attr


def test_twelve_to_twenty_quarter_figures_carry_the_anecdote_label():
    """12-20Q figures always carry 「只能当个案」 (§10.6, U-13)."""
    for name in OWNED:
        lines = src(name).splitlines()
        for i, line in enumerate(lines):
            if re.search(r"12[–-]20 个季度|(?<![前第\d])(?:12|20) 个季度|\b(?:12|20)Q\b", line) and not line.lstrip().startswith("//"):
                window = " ".join(lines[max(0, i - 1):i + 3])
                assert "只能当个案" in window or "ANECDOTE" in window, (name, i + 1, line.strip())


def test_badges_and_engine_labels():
    """Every stated or completed field is labelled (文字数字 / 文字定性 / 推算 / 模型补全 / 锚 / 干预), and the quarterly
    stack is labelled 试运行 wherever it is shown."""
    qd = src("_qdetail.js")
    assert "export const BADGE = { stated: '文字数字', stated_qual: '文字定性', derived: '推算', completed: '模型补全', anchor: '锚', intervention: '干预' };" in qd
    assert "export const TRIAL = '试运行';" in qd
    assert qview.BADGES == {"stated": "文字数字", "stated_qual": "文字定性", "derived": "推算", "completed": "模型补全",
                            "anchor": "锚", "intervention": "干预"}
    for name in ("forecast.js", "_qlab.js", "_scenario_detail.js"):
        assert "TRIAL" in src(name), name


def test_real_translations_need_an_explicit_tick():
    """The lab's translate button: a 409 from the server shows the estimate and a tick box before confirm: true."""
    qlab = src("_qlab.js")
    assert "if (res.status === 409 && body.estimate)" in qlab
    assert "'我确认：这会计入我的 Claude 订阅用量。'" in qlab
    assert "...(confirm ? { confirm: true } : {})" in qlab
    assert "tick.addEventListener('change', () => { go.disabled = !tick.checked; });" in qlab


# ================================================================ the mixture (scenario_mix.py; forecast.js mixQuantile)

def test_mixture_quantiles_match_monte_carlo():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.0, 1.0, 20000), rng.normal(4.0, 0.5, 20000)
    ga, gb = SM.quantile_grid(a), SM.quantile_grid(b)
    pool = np.concatenate([a[:15000], b[:5000]])                 # weights 0.75 / 0.25
    for q in (0.1, 0.5, 0.9):
        got = SM.mix_quantile([ga, gb], [0.75, 0.25], q)
        assert abs(got - np.quantile(pool, q)) < 0.06, q
    # one distribution: its own quantiles back (to the grid's interpolation)
    assert abs(SM.mix_quantile([ga], [1.0], 0.5) - np.median(a)) < 1e-6
    assert abs(SM.mix_cdf([ga, gb], [3, 1], float(np.median(a))) - 0.375) < 0.01
    # unavailable scenarios (no grid, zero weight) are skipped
    assert abs(SM.mix_quantile([ga, None, gb], [0.75, 0.5, 0.25], 0.5) - SM.mix_quantile([ga, gb], [0.75, 0.25], 0.5)) < 1e-12
    assert SM.mix_quantile([None], [1.0], 0.5) is None
    assert SM.quantile_grid(np.full(10, np.nan)) is None


def test_the_page_mixes_with_the_same_rule():
    """forecast.js's gridCdf / mixQuantile are the scenario_mix rule (piecewise linear CDF, exact inversion over the
    union of the knots)."""
    fc = src("forecast.js")
    assert "export function gridCdf(g, P, x)" in fc and "export function mixQuantile(grids, weights, P, q)" in fc
    assert "return hi > lo ? p0 + (p1 - p0) * ((x - lo) / (hi - lo)) : p1;" in fc
    assert "if (F(knots[mid]) >= q) hi = mid; else lo = mid + 1;" in fc
    assert "return f1 > f0 ? x0 + ((q - f0) * (x1 - x0)) / (f1 - f0) : x1;" in fc


def test_the_mixture_is_exact_with_a_divergent_path():
    """A grid whose maximum is astronomically large (a divergent path) does not move the mixture's central quantiles:
    the inversion is exact on the piecewise-linear CDF."""
    rng = np.random.default_rng(5)
    a = rng.normal(0.0, 0.1, 5000)
    b = np.append(rng.normal(0.05, 0.1, 4999), 1e45)
    ga, gb = SM.quantile_grid(a), SM.quantile_grid(b)
    assert gb[-1] == pytest.approx(1e45)
    pool = np.concatenate([a, b[:-1]])
    for q in (0.1, 0.5, 0.9):
        assert abs(SM.mix_quantile([ga, gb], [0.5, 0.5], q) - np.quantile(pool, q)) < 0.01, q
    # the quantile is the inverse of the CDF
    for q in (0.05, 0.37, 0.95):
        x = SM.mix_quantile([ga, gb], [0.3, 0.7], q)
        assert SM.mix_cdf([ga, gb], [0.3, 0.7], x) == pytest.approx(q, abs=1e-9)


def test_grid_cdf_and_crps():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 50000)
    g = SM.quantile_grid(x)
    for v in (-1.0, 0.0, 1.5):
        assert abs(SM.grid_cdf(g, v) - (0.5 * (1 + math.erf(v / math.sqrt(2))))) < 0.01
    assert SM.grid_cdf(g, -99) == 0.0 and SM.grid_cdf(g, 99) == 1.0
    # CRPS of N(0,1) at 0 is (sqrt(2) - 1) / sqrt(pi) ~ 0.2337
    assert abs(SM.crps_grid(g, 0.0) - (math.sqrt(2) - 1) / math.sqrt(math.pi)) < 0.01
    assert SM.crps_grid(g, 3.0) > SM.crps_grid(g, 0.0)


def test_weights_renormalise_over_the_scenarios_with_a_result():
    w = SM.weights_for({"a": 0.5, "b": 0.3, "c": 0.2}, ["a", "b", "c"], {"a": True, "b": False, "c": True})
    assert set(w) == {"a", "c"} and abs(w["a"] - 0.5 / 0.7) < 1e-12 and abs(sum(w.values()) - 1) < 1e-12


def _mix_fixture():
    rng = np.random.default_rng(3)
    grid = lambda m: [SM.quantile_grid(rng.normal(m * (k + 1), 0.05, 4000)) for k in range(4)]   # noqa: E731
    eng = {"version": "test/1", "scenarios": {"s1": {"assets": {"us_equity": {"grid": grid(0.01)}}},
                                              "s2": {"assets": {"us_equity": {"grid": grid(-0.02)}}}}}
    return {"created": "2026-11-02T00:00:00Z", "origin": "2026Q4", "quarters": ["2026Q4", "2027Q1", "2027Q2", "2027Q3"],
            "assets": ["us_equity"], "engines": {"monthly": eng}, "default_engine": "monthly",
            "scenarios": [{"id": "s1"}, {"id": "s2"}], "versions": {"card": "c1"},
            "probabilities": {"default": "prior", "sets": {"prior": {"probs": {"s1": 0.7, "s2": 0.3}}}, "version": "p1"}}


def test_the_ledger_is_append_only_and_scores_arrived_quarters(tmp_path):
    mix = _mix_fixture()
    e = SM.ledger_entry(mix, "monthly")
    assert e["key"] and len(e["assets"]["us_equity"]) == 4 and e["probs"] == {"s1": 0.7, "s2": 0.3}
    path = tmp_path / "ledger" / "forecast_ledger.jsonl"
    assert SM.append_ledger(path, [e]) == {"appended": 1, "skipped": 0}
    assert SM.append_ledger(path, [e]) == {"appended": 0, "skipped": 1}           # same versions: not again
    other = dict(e, key="other", probabilities="p2")
    assert SM.append_ledger(path, [other])["appended"] == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["key"] == e["key"]          # the first line never changes
    path.write_text(path.read_text(encoding="utf-8") + "{torn line\n", encoding="utf-8")
    assert len(SM.read_ledger(path)) == 2                                        # a torn line is skipped
    # two quarters have arrived: realized +1% and +0.5% (log total returns in the quarterly panel)
    Q = pd.DataFrame({"yt_us_equity": [0.01, 0.005, np.nan, np.nan]}, index=["2026Q4", "2027Q1", "2027Q2", "2027Q3"])
    assert SM.realized_cum(Q, e["quarters"], "us_equity")[:2] == pytest.approx([math.expm1(0.01), math.expm1(0.015)])
    assert SM.realized_cum(Q, e["quarters"], "us_equity")[2:] == [None, None]
    sc = SM.score_ledger([e], Q)
    rec = sc["entries"][0]
    assert rec["matured"] == 2 and 0 <= rec["coverage80"] <= 1 and rec["crps"] > 0
    assert sc["summary"]["monthly"]["n"] == 2


# ================================================================ the QTable display view (qview.py)

APPENDIX_B = {"schema": "qtable/1", "n_quarters": 4, "rows": [
    {"q": 1, "x": {"d_oil": {"value": 26, "sd": 8, "src": "stated", "quote": "油价一个季度内上涨约三成"},
                   "x_oil_cause": {"value": "supply", "src": "stated_qual", "quote": "产油国卷入战争，供应中断"},
                   "d_gpr": {"sign": "+", "size": "大幅", "src": "stated_qual", "quote": "地缘政治风险急剧上升"},
                   "d_cpi_goods": {"sign": "+", "size": "明显", "src": "stated_qual", "quote": "商品价格明显上涨"}},
     "fc": {"stance": "收紧", "quote": "信用利差走阔"}},
    {"q": 2, "x": {"d_ff": {"value": 0.5, "src": "stated", "quote": "央行加息 50 个基点"}}, "fc": {"stance": "未说明"}},
    {"q": 3, "x": {}, "u": {"u_ff": {"sign": "-", "size": "明显", "src": "stated_qual", "quote": "政治压力下央行意外降息"}},
     "fc": {"stance": "中性", "quote": "金融条件大体平稳"}},
    {"q": 4, "x": {"d_unrate": {"value": 0.2, "src": "derived", "quote": "失业率一年升约 0.8"}}, "fc": {"stance": "未说明"}}],
    "c_q": {"anchors": {"r_star": {"value": 0.75, "from_q": 2, "quote": "中性利率维持在目前的估计"}},
            "constraints": [{"kind": "range", "field": "d_unrate", "quarters": [1, 2, 3], "min": -0.2, "max": 0.3,
                             "quote": "就业基本稳定"}],
            "regime": [{"quarters": [3, 4], "label": "recession", "share": 0.6, "quote": "经济陷入衰退"}]}}


def test_the_view_labels_every_channel_quarter_by_quarter():
    rows = qview.cells(APPENDIX_B)
    assert [r["q"] for r in rows] == [1, 2, 3, 4]
    c1, c2, c3, c4 = (r["cells"] for r in rows)
    assert [i["badge"] for i in c1["oil"]["items"]] == ["stated", "stated_qual"]
    assert c1["inflation"]["items"][0]["badge"] == "stated_qual" and "明显" in c1["inflation"]["items"][0]["text"]
    assert c1["fc"]["items"][0]["badge_zh"] == "文字定性" and c1["fc"]["stance"] == "收紧"
    assert c2["fc"]["completed"] and c2["fc"]["stance"] == "未说明"
    assert c2["policy"]["items"][0]["badge"] == "stated" and "（硬约束）" in c2["policy"]["items"][0]["text"]
    assert c3["policy"]["items"][0]["badge"] == "intervention" and c3["policy"]["items"][0]["text"].startswith("干预")
    assert c4["growth"]["items"][0]["badge"] == "derived"
    assert any(i.get("constraint") for i in c1["growth"]["items"]) and not any(i.get("constraint") for i in c4["growth"]["items"])
    assert any("衰退" in i["text"] for i in c3["growth"]["items"])
    assert c1["valuation"]["completed"] and c1["valuation"]["mode"] == "inferred"
    assert [i["text"] for i in rows[0]["other"]] and rows[0]["other"][0]["badge"] == "stated_qual"   # d_gpr
    assert rows[0]["anchors"] == [] and rows[1]["anchors"][0]["badge"] == "anchor"
    assert c1["rates"]["completed"] and c1["policy"]["completed"]
    # quotes travel with the items, and no text shows a code id
    assert c1["oil"]["items"][0]["quote"] == "油价一个季度内上涨约三成"
    texts = [i["text"] for r in rows for c in r["cells"].values() for i in c["items"]] + \
            [i["text"] for r in rows for i in r["other"] + r["anchors"]]
    assert texts and not [t for t in texts if re.search(r"\b[dsux]_[a-z0-9_]+\b|r_star", t)]
    assert [c["key"] for c in qview.channels()] == ["policy", "growth", "inflation", "oil", "fc", "rates", "valuation"]


# ================================================================ the valuation view (valview.py)

def _synthetic_panel(n=40):
    idx = pd.period_range("1973Q1", periods=n, freq="Q").astype(str)
    rng = np.random.default_rng(4)
    mult, fund = rng.normal(0.004, 0.06, n), rng.normal(0.01, 0.01, n)
    return pd.DataFrame({"y_us_equity": mult + fund, "y_mult": mult, "y_fund": fund, "y_pd": mult * 0.9,
                         "y_eps": rng.normal(0.01, 0.05, n), "s_cape": np.log(np.linspace(10, 30, n)),
                         "s_pd": np.log(np.linspace(30, 90, n)), "s_pe": np.log(np.linspace(9, 25, n)),
                         "s_real_y10": np.full(n, 1.5), "s_fcfy": np.linspace(5, 2, n), "s_pb_buseq": np.zeros(n),
                         "s_ecyc": np.zeros(n)}, index=idx)


def test_the_valuation_view_is_the_identity():
    from scenariolab.qmodel import valview
    Q = _synthetic_panel()
    v = valview.view(Q, {"data_end_month": "1982-09"})
    assert v["identity"]["max_abs_error"] == pytest.approx(0.0, abs=1e-12)
    assert v["identity"]["all"]["mult"]["corr"] is not None
    cape = next(s for s in v["today"]["states"] if s["key"] == "cape")
    assert cape["value"] == pytest.approx(30.0, rel=1e-6) and cape["percentile"] == pytest.approx(1.0)
    assert len(v["cape_q"]) == 101 and v["cape_q"][0] == pytest.approx(10.0, rel=1e-6)
    assert any("相当于直接假设股价" in n for n in v["notes"]) and any("纳指的估值只能用全市场 CAPE 近似" in n for n in v["notes"])
    assert any("不是 SPY / QQQ 的自由现金流收益率" in n for n in v["notes"])


@needs_panel
def test_the_valuation_view_reproduces_the_spec_check_values():
    """§6.1 table, §4.4 re-rating drift and §6.2 states (CAPE 41.2, P/D 97.4, FCF 1.96%, BusEq P/B 12.2 and 15.1)."""
    r = client.get("/api/q/valuation")
    assert r.status_code == 200
    v = r.json()
    win = {w["from"]: w for w in v["identity"]["windows"]}
    for start, vals in {"2007Q3": (-0.468, -0.492, -0.460, 1.005), "2001Q1": (-0.383, -0.469, -0.357, 0.256),
                        "1995Q3": (1.008, 0.668, 0.789, 0.621), "2022Q1": (-0.296, -0.408, -0.346, -0.457)}.items():
        w = win[start]
        assert [w["price"], w["mult"], w["pd"], w["pe"]] == pytest.approx(list(vals), abs=0.0015), start
    assert v["identity"]["all"]["mult"]["corr"] == pytest.approx(0.990, abs=0.002)
    assert v["identity"]["all"]["mult"]["share"] == pytest.approx(1.011, abs=0.002)
    assert [x["per_year"] for x in v["rerating"]][:4] == pytest.approx([0.0118, 0.0180, 0.0147, 0.0189], abs=0.0002)
    st = {s["key"]: s for s in v["today"]["states"]}
    assert st["cape"]["value"] == pytest.approx(41.2, abs=0.05)
    assert st["pd"]["value"] == pytest.approx(97.4, abs=0.05)
    assert st["fcf"]["value"] == pytest.approx(1.96, abs=0.01)
    assert st["pb_buseq"]["value"] == pytest.approx(12.2, abs=0.05) and st["pb_buseq"]["ref"]["value"] == pytest.approx(15.1, abs=0.05)


# ================================================================ endpoints

def test_the_scenario_mix_endpoint_names_its_build_step(monkeypatch, tmp_path):
    from scenariolab import server
    monkeypatch.setattr(server, "RESULTS", tmp_path)
    r = client.get("/api/forecast/scenario_mix")
    assert r.status_code == 404
    assert "--only forecast" in r.json()["detail"] and "scenario_mix" in r.json()["detail"]
    (tmp_path / "forecast_mix.json").write_text('{"version": 1}', encoding="utf-8")
    assert client.get("/api/forecast/scenario_mix").json() == {"version": 1}
    assert client.get("/api/forecast_mix").json() == {"version": 1}            # the generic result route too
    assert "forecast_mix" in server.RESULT_NAMES


def test_the_forecast_build_step_writes_the_mixture():
    """§3.9: the forecast build step gains the scenario mixture (and the forward ledger) after forecast.json."""
    import inspect

    from scenariolab import build
    assert 'step("forecast", lambda: build_forecast(ctx, out))' in inspect.getsource(build.main)
    body = inspect.getsource(build.build_forecast)
    assert body.index('"forecast.json"') < body.index('_run("analysis.scenario_mix", ctx, results=out)')
    assert SM.OUTPUT == "forecast_mix.json"


@needs_panel
def test_a_quarterly_run_carries_what_the_detail_shows():
    """/api/q/run's result (api.execute) carries the per-quarter cells, the valuation statements' implied moves, the
    attribution at 4 and 40 quarters and the regime readout on quarterly paths."""
    from scenariolab.qmodel import api as qapi
    qt = json.loads(json.dumps(APPENDIX_B))
    qt["n_quarters"] = 8
    qt["rows"] += [{"q": q, "x": {}, "fc": {"stance": "未说明"}} for q in range(5, 9)]
    qt["c_q"]["valuation"] = [{"quarters": [1, 2, 3, 4], "mode": "rerate", "direction": "-", "size": "明显",
                              "quote": "估值压缩"}]
    narrative = "；".join(sorted({e["quote"] for r in qt["rows"] for e in list(r.get("x", {}).values())
                                 + list((r.get("u") or {}).values()) + [r["fc"]] if e.get("quote")}
                                | {"中性利率维持在目前的估计", "就业基本稳定", "经济陷入衰退", "估值压缩"}))
    req = qapi.parse_request({"qtable": qt, "narrative": narrative, "source": "user", "n_paths": 200, "seed": 1,
                              "mode": "path", "continuation": 40})
    res = qapi.execute(req)
    assert len(res["cells"]) == 8 and [c["key"] for c in res["channels"]][0] == "policy"
    vs = res["valuation_statements"]
    assert len(vs) == 1 and vs[0]["n"] == 4 and vs[0]["mult_p50"] < 0
    assert set(res["attribution_by_h"]) == {"4"}                              # an 8-quarter scenario: 4 < H < 40
    assert {"stated", "completed", "drift", "valuation", "residual"} <= set(res["attribution_by_h"]["4"]["us_equity"])
    assert res["regimes"]["years"] and abs(sum(v or 0 for v in res["regimes"]["years"][0]["mix"].values()) - 1) < 0.02
    assert set(res["y_fund"]) == {"p10", "p50", "p90"}


@needs_paths
def test_the_mixture_builds_from_the_stored_paths_and_the_stack():
    """scenario_mix.run without writing: both engines, every built-in's statements for four quarters, the default
    probabilities summing to 1, the base rates and the unconditional bands."""
    mix = SM.run(None, n_paths=200, seed=3, ledger=False, write=False)
    assert mix["default_engine"] == "monthly" and set(mix["engines"]) == {"monthly", "quarterly"}
    assert mix["engines"]["quarterly"]["trial"] is True and mix["switch_gate"]["status"] == "not_evaluated"
    assert len(mix["quarters"]) == 4 and len(mix["grid"]) == len(SM.GRID)
    probs = mix["probabilities"]["sets"]["prior"]["probs"]
    assert abs(sum(probs.values()) - 1) < 1e-6 and set(probs) == {s["id"] for s in mix["scenarios"]}
    for e in mix["engines"].values():
        for sid, blk in e["scenarios"].items():
            if blk.get("assets") and (blk.get("finite") or 0) >= 0.5:
                g = blk["assets"]["us_equity"]["grid"][3]
                assert g is not None and all(a <= b for a, b in zip(g, g[1:])), sid
    assert all(len(v) == 4 for v in mix["statements"].values())
    for k in ("contraction_4q", "cpi_gt4_q4", "ff_up1_q4"):
        assert 0 <= mix["history_rates"][k] <= 1
    assert mix["engines"]["quarterly"]["reference"]["assets"]["us_equity"]["p50"][3] is not None
    text = json.dumps(mix, ensure_ascii=False)
    assert "SIM_DESIGN" not in text and "P5" not in text                       # no document or phase ids for readers


# ================================================================ review fixes: what each engine ran, readable prose

READER_TABS = ["forecast.js", "_qdetail.js", "evaluation.js", "_switch.js", "overview.js", "scenarios.js",
               "portfolio.js", "withdrawal.js", "method.js"]
CODE_ID = re.compile(r"(?<![A-Za-z0-9_./-])[a-z]{1,6}_[a-z0-9_]+(?![A-Za-z0-9_(])|\.jsonl?\b|§\s?\d|SIM_DESIGN")
BUILD_HINT = ("还没有生成", "还没有", "暂缺", "重启", "重新运行", "尚无")


def _reader_literals(text: str) -> list[str]:
    """The Chinese string literals of a module, without their ${…} expressions and the comment lines."""
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    out = []
    for m in re.finditer(r"'((?:[^'\\\n]|\\.)*)'|`((?:[^`\\]|\\.)*)`", text):
        t = m.group(1) if m.group(1) is not None else m.group(2)
        if re.search(r"[一-鿿]", t):
            out.append(re.sub(r"\$\{[^}]*\}", "", t))
    return out


def test_reader_prose_carries_no_code_ids_file_names_or_section_references():
    """§10.6 / the UI conventions: Chinese prose shows no code ids, file names, § references or the spec's name; a
    build hint for a missing result (the app's convention) may name its command and file."""
    bad = []
    for name in READER_TABS:
        for t in _reader_literals(src(name)):
            if any(w in t for w in BUILD_HINT) and ("python -m" in t or ".json" in t):
                continue
            if CODE_ID.search(t):
                bad.append((name, t[:80]))
    assert not bad, bad


def test_section_3_shows_what_the_selected_engine_ran():
    """Under the monthly engine the section lists the scenario library's own phase specs (views, overrides, the CMA's
    valuation reversion) and the records the QTable could not carry; the QTable panel is the quarterly stack's."""
    fc = src("forecast.js")
    assert "两个引擎读同一份" not in fc
    assert "月度引擎运行的是情景库原来的阶段规格" in fc and "季度情景表没法写进的" in fc
    assert "loadMonthlySpec(ctx.api, sid)" in fc and "api.get(`q/scenarios/${sid}`)" in fc
    assert "asset_drift" in fc and "资产漂移覆盖" in fc and "VALUATION_VIEWS" in fc and "cma.reversion" in fc
    assert "你必须相信什么（月度引擎运行的阶段规格）" in fc and "你必须相信什么（季度情景表写下的陈述）" in fc
    # the headline counts the scenarios the engine includes; excluded ones are named in the overview table
    assert "ids.filter(usable).length" in fc and "未计入（没有可用路径）" in fc


def test_the_quarterly_set_never_sends_tickers():
    """On a set with step_months 3 the ticker rows are disabled and the request carries the asset classes only (the
    server refuses tickers there); a ticker-only portfolio sends nothing and says why."""
    tk = src("_tickers.js")
    assert "function lockTickers(reason)" in tk and "function getAssets()" in tk and "TICKERS_Q_ONLY" in tk
    for name in ("portfolio.js", "withdrawal.js"):
        s = src(name)
        assert "editor.lockTickers(quarterlySet() ? TICKERS_Q : null);" in s, name
        assert "const reqWeights = () => (quarterlySet() ? editor.getAssets() : editor.get());" in s, name
        assert "const weights = reqWeights();" in s and "TICKERS_Q_ONLY" in s, name


def test_one_probability_basis_and_the_live_switch():
    """The hand-set probability is the default basis on the overview and scenarios tabs (the tilt is a labelled
    comparison); the overview's mixture follows the live switch, not the build-time copy."""
    ov, sc = src("overview.js"), src("scenarios.js")
    assert "loadSwitch()" in ov and "quarterlyDefault(sw)" in ov and "mix.default_engine" not in ov
    assert "手设概率（默认）" in ov and "倾斜后（对照）" in ov
    assert "const pDef = (s) => (isNum(s.prior_prob) ? s.prior_prob : 0);" in sc and "按后验概率" not in sc
    assert "不是预测。'" not in sc and "透明的启发式" not in src("method.js")
    assert '"default_engine": "quarterly" if gate["passed"] else ("monthly" if "monthly" in engines else None)' in \
        (config.ROOT / "scenariolab" / "analysis" / "scenario_mix.py").read_text(encoding="utf-8")


def test_detail_wording_glitches_are_fixed():
    """The delivery table reads 精确满足 / 个标准差 / the oil-cause words; the valuation block says the legs are
    separate medians and writes the horizon in quarters; clip notes name the field; the version string is collapsed."""
    qd = src("_qdetail.js")
    assert "toExponential" not in qd and "±1 sd" not in qd and "精确满足" in qd and "供给冲击" in qd
    assert "不能直接相加" in qd and "Math.round((k + 1) / 4)" not in qd and "约 ${fx((k + 1) / 4" in qd
    assert "noteText(n)" in qd and "'给维护者'" in qd and "美元保持项" not in qd
    ev = src("evaluation.js")
    assert "流程检验（不作结论）" in ev and "模拟探针总回答“不知道”，不是测量" in ev and "function maintainer(" in ev
    fc = src("forecast.js")
    assert "'预测区间'" not in fc and "'相似起点法区间'" in fc and "预测本身用的是" not in fc


def test_server_reader_texts_carry_no_code_ids():
    """The texts the server hands to the pages: the switch's not-evaluated text, the SEP arm's status, the valuation
    notes and the evaluation tab's notes."""
    from scenariolab.evaluation import switch as SW
    from scenariolab.evaluation.narrative import sep as SEP
    from scenariolab.qmodel import valview
    texts = [SW.not_evaluated_text(), SW.MOCK_RUN_TEXT, SEP.NEEDS_DATA, *valview.NOTES]
    for t in texts:
        assert not CODE_ID.search(t) and "python -m" not in t, t
    assert "智能体推演" not in SW.not_evaluated_text()               # the agent tree runs the stub by default
    assert all("预测的" not in t and "实际对预测" not in t for t in valview.NOTES)
    assert any("只能当个案" in t and "1–20 个季度" in t for t in valview.NOTES)
