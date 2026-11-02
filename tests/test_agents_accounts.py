"""The agent tree's accounts (scenariolab/agents/accounts.py; LEDGERS.md, AGENT_DESIGN §5): the per-particle books,
the identities I1–I10, the gates G-L1…G-L9, the s-field views (I9), the observation packets, the build step and
its cache. No network, no LLM: a synthetic root snapshot and a seeded Gaussian stub stand in for the data and the
quarterly model. Tests marked `needs_cache` also check the real cache when the research import has filled it."""
import json

import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.agents import accounts as AC
from scenariolab.agents import actors as A

N = 300


def _line(v, unit="$bn", as_of="2026-10-28"):
    return {"value": v, "unit": unit, "as_of": as_of, "src": "TEST", "series": "TEST"}


def synthetic_root() -> dict:
    """A root snapshot with every line the books need (rounded, synthetic test values)."""
    books = {
        "macro": {"gdp_saar": 33000.0, "y2": 4.6, "y10": 5.0, "psavert_p1_since_1959": 2.3},
        "fed": {"iorb": 3.9, "soma_treasuries": 4577.0, "soma_bills": 572.0, "soma_mbs": 1884.0, "total_assets": 6746.0,
                "currency": 2485.0, "reverse_repo_total": 326.0, "on_rrp": 4.4, "tga": 1016.0, "deferred_asset": 232.5,
                "other_liabilities_capital_resid": 258.4, "soma_bills_change_13w": 57.0, "soma_mbs_change_13w": -47.1,
                "net_income_last4q": 10.2, "deferred_asset_peak": 244.2, "core_pce_yoy": 3.1, "inflation_gap": 1.1,
                "unemployment": 4.2, "nrou_cbo": 4.39},
        "fiscal": {"debt_held_public": 32712.0, "intragov_holdings": 7881.0, "statutory_limit": 41104.0,
                   "not_subject_to_limit": 202.9, "limit_headroom": 713.0, "avg_rate_marketable": 3.52,
                   "avg_rate_marketable_12m_ago": 3.41, "foreign_holdings_tic": 9332.0, "deficit_nipa_gdp": 5.2,
                   "nipa_interest_saar": 1329.0, "debt_held_public_nipa_q": 31706.0, "interest_burden_nipa": 4.03,
                   "goods_imports_saar": 4020.0, "receipts_q": 1252.0, "intragov_growth_q": 73.2, "debt_pace_6m": 274.0,
                   "months_to_limit_at_6m_pace": 2.6, "receipts_fy": 5403.0, "outlays_fy": 7396.0, "deficit_fy": 1993.0,
                   "net_interest_fy": 1052.0, "net_interest_share_receipts": 19.5, "debt_gdp": 97.4,
                   "customs_refunds_since_2026_05": 134.0, "customs_gross_3m": 73.5, "tga_dts": 1040.0,
                   "mspd_bills": 7118.0, "mspd_notes": 16282.0, "mspd_bonds": 5551.0, "bill_share": 22.4},
        "president": {"effective_tariff_rate": 6.85, "gasoline_price": 4.27, "cpi_yoy": 3.7, "us_equity_drawdown": -0.8,
                      "real_dpi_pc_yoy": 0.7, "spr_inventory": 283.0, "spr_change_52w": -127.0},
        "china": {"fx_reserves_ex_gold": 3482.0, "us_treasury_holdings": 591.0, "cny_per_usd": 6.67,
                  "private_credit_gdp": 201.0, "gg_net_lending_gdp_2026": -8.0, "gg_gross_debt_gdp_2026": 104.0,
                  "real_gdp_growth_2026": 4.5, "real_house_price_vs_peak": -24.7, "us_imports_from_china_12m": 273.0},
        "opec": {"brent_spot": 123.0, "saudi_reserves_ex_gold": 487.0, "us_crude_production": 14.0,
                 "sau_gg_net_lending_gdp_2026": -6.5, "sau_real_gdp_growth_2026": -3.3, "kwt_real_gdp_growth_2026": -16.7,
                 "irq_real_gdp_growth_2026": -16.0, "are_real_gdp_growth_2026": -3.9},
        "bigtech": {"revenue": 506.0, "capex": 230.0, "depreciation": 50.7, "dividends": 9.7, "buybacks": 4.4,
                    "lease_principal": 3.5, "debt": 548.0, "cash": 180.0, "securities": 383.0, "cfo": 199.0, "fcf": -35.3,
                    "operating_income": 134.0, "capex_ttm": 691.0, "capex_ttm_prev": 357.0, "payouts_ttm": 70.3,
                    "debt_change_4q": 309.0, "net_cash": 15.8, "revenue_growth_4q": 18.6},
        "households": {"equities_direct": 40000.0, "mutual_funds": 26500.0, "real_estate": 54100.0,
                       "liabilities": 21900.0, "net_worth": 195900.0, "pce_saar": 21500.0, "saving_rate": 3.75,
                       "dpi_saar": 24300.0, "debt": 21400.0, "equity_share_fin_assets": 43.3, "nw_share_top1": 32.5,
                       "nw_share_90_99": 36.4, "nw_share_50_90": 28.8, "nw_share_bottom50": 2.3},
        "nfc": {"nfc_debt": 15700.0, "nfc_pbt": 3500.0, "nfc_net_interest": 364.0, "nfc_capex": 3250.0,
                "interest_coverage": 10.6},
        "banks": {"bank_total_assets": 26460.0, "bank_loans": 14220.0, "bank_securities": 5800.0,
                  "bank_deposits": 19770.0, "bank_chargeoff": 0.55, "bank_nii_q": 197.0},
        "investors": {"margin_debt": 1480.0, "margin_debt_yoy": 31.4},
        "row": {"niip": -22400.0, "current_account_q": -246.0},
        "sectoral": {"nl_nation_row_gdp": -3.35, "nl_government": -2280.0, "nl_federal": -1830.0, "nl_private_gdp": 3.6},
    }
    root = {"version": AC.ACCOUNTS_VERSION, "as_of_run": "2026-11-01",
            "books": {b: {"lines": {k: _line(v) for k, v in lines.items()}, "history": {}} for b, lines in books.items()}}
    root["books"]["macro"]["history"]["y10_q_20"] = {f"q{i}": 3.0 + 0.1 * i for i in range(20)}
    root["books"]["fed"]["history"]["reserves"] = {"2025Q4": 2994.0, "2026Q1": 2951.0, "2026Q2": 2948.0, "2026Q3": 2894.0}
    root["books"]["bigtech"]["capex_history"] = {"quarters": [f"q{i}" for i in range(37)],
                                                 "values": list(np.linspace(20, 230, 37))}
    return root


def stub_draws(n: int, quarters: int, seed: int = 11, **means) -> list[tuple[dict, dict]]:
    """A seeded Gaussian stand-in for the quarterly model's joint draw (schema ids, plausible scales)."""
    rng = np.random.default_rng(seed)
    spec = {"d_ff": (0, 0.0), "d_core": (3.0, 1.0), "d_unrate": (0.0, 0.3), "d_deficit": (0.0, 0.6),
            "d_y2": (0.0, 0.3), "d_y10": (0.0, 0.3), "d_cpi_goods": (2.0, 5.0), "d_cpi_serv": (3.5, 1.2),
            "d_rgdp": (2.5, 2.0), "d_cons_goods": (3.0, 4.0), "d_cons_serv": (2.5, 1.5), "d_hpi": (3.0, 5.0),
            "d_profits": (1.0, 7.0), "d_capex": (4.0, 7.0), "d_itinv": (10.0, 9.0), "d_chargeoff": (0.0, 0.1),
            "x_lending_std": (5.0, 10.0), "d_tariff": (0.0, 0.13), "d_usd": (0.0, 3.0), "d_oil": (0.0, 15.0),
            "d_bdlev": (0.0, 5.0)}
    out = []
    for _ in range(quarters):
        x = {k: rng.normal(means.get(k, m), s, n) for k, (m, s) in spec.items()}
        y = {"us_equity": rng.normal(0.02, 0.08, n), "treasury_10y": rng.normal(0.01, 0.04, n),
             "cash": np.full(n, 0.009), "crude_oil": rng.normal(0.0, 0.15, n)}
        out.append((x, y))
    return out


def run_path(actions: list[dict], n: int = N, seed: int = 11):
    acc = AC.Accounts.from_root(synthetic_root(), n, institutions={"em_capacity_bn": 350.0, "tga_target_bn": 850.0})
    reports, gates = [], []
    for (x, y), a in zip(stub_draws(n, len(actions), seed), actions):
        reports.append(acc.step(x, y, a))
        gates.append(AC.gates(acc))
    return acc, reports, gates


def paths():
    base = [{} for _ in range(8)]
    base[1] = A.account_actions([A.option_payload("bigtech", "capex_raise_large", 15)])
    a = [dict(x) for x in base]
    a[2] = {**a[2], **A.account_actions([A.option_payload("congress", "debt_limit_raise", 3000)])}
    b = [dict(x) for x in base]
    c = [dict(x) for x in a]
    c[1] = A.account_actions([A.option_payload("bigtech", "capex_slash", -35), A.option_payload("bigtech", "buyback_up")])
    return {"A": a, "B": b, "C": c}


# ---------------------------------------------------------------- the catalogue of books

def test_the_line_catalogue_has_fifteen_books_and_known_update_classes():
    assert len(AC.BOOKS) == 15 and len(AC.LINES) == 126
    assert {r[5] for r in AC.LINES} <= set(AC.UPDATE_CLASSES)
    assert {r[0] for r in AC.LINES} <= set(AC.BOOKS)
    ids = [(r[0], r[1]) for r in AC.LINES]
    assert len(ids) == len(set(ids))
    assert len(AC.lines("fed")) == 17 and AC.lines("fed")[0]["id"] == "total_assets"
    assert {f"A{i}" for i in range(1, 17)} == set(AC.CALIBRATIONS)
    assert set(AC.IDENTITIES) == {f"I{i}" for i in range(1, 11)}
    assert len(AC.ACCOUNT_INPUTS) == 21


def test_account_inputs_resolve_through_the_alias_table():
    r = A.FieldResolver(fields=["d_ff", "d_core", "d_y10", "d_lf"], aliases={"d_policy": "d_ff"})
    inp = AC.resolve_inputs(r)
    assert inp["ff"] == "d_ff" and inp["y10"] == "d_y10"
    acc = AC.Accounts.from_root(synthetic_root(), 5, resolver=r)
    x, y = stub_draws(5, 1)[0]
    x["d_policy"] = x.pop("d_ff") + 0.25          # an old id is still read through the alias
    rep = acc.step(x, y, {})
    assert rep["ok"] and np.allclose(acc.s["iorb"], 3.9 + x["d_policy"])


# ---------------------------------------------------------------- T-L1 identities on three action paths

@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_identities_hold_every_quarter_on_three_action_paths(name):
    acc, reports, _ = run_path(paths()[name])
    assert all(r["ok"] for r in reports), [r["residuals"] for r in reports if not r["ok"]]
    worst = max(max(r["residuals"].values()) for r in reports)
    assert worst < 1e-6
    assert all(not r["inputs_missing"] for r in reports)
    assert acc.quarter == "2028Q3" and acc.q == 8


def test_missing_model_inputs_are_reported_not_fatal():
    acc = AC.Accounts.from_root(synthetic_root(), 10)
    x, y = stub_draws(10, 1)[0]
    del x["d_bdlev"], x["d_chargeoff"]
    rep = acc.step(x, y, {})
    assert rep["ok"] and set(rep["inputs_missing"]) == {"d_bdlev", "d_chargeoff"}


# ---------------------------------------------------------------- T-L2 negative tests

@pytest.mark.parametrize("label,key,delta,identity", [
    ("reserves +1bn without a counterpart", "res", 1.0, "I1_fed_balance_sheet"),
    ("household saving dropped", "hh_ofa", -10.0, "I6_household_stock_flow"),
    ("hyperscaler cash leak", "bt_cash", 5.0, "I7_company_cash"),
    ("public debt booked twice", "dhp", 100.0, "I4_public_debt"),
    ("TGA moved without borrowing", "tga", 7.0, "I3_tga_cash"),
    ("bank securities without a counterpart", "bk_sec_cost", 3.0, "I10_banks"),
])
def test_each_identity_catches_a_broken_flow(label, key, delta, identity):
    acc = AC.Accounts.from_root(synthetic_root(), 20)
    x, y = stub_draws(20, 1)[0]
    acc.step(x, y, {})
    assert acc.ok()
    acc.s[key] = acc.s[key] + delta
    acc.check()
    assert not acc.ok(), label
    assert acc.resid[identity] > 1e-6, (label, acc.resid)


# ---------------------------------------------------------------- particles move with their books

def test_take_and_concat_move_books_with_particles_without_averaging():
    acc = AC.Accounts.from_root(synthetic_root(), 40)
    for x, y in stub_draws(40, 3, seed=5):
        acc.step(x, y, {})
    idx = np.array([3, 3, 7, 0])
    sub = acc.take(idx)
    assert sub.n == 4 and np.array_equal(sub.s["tga"], acc.s["tga"][idx])
    assert np.array_equal(sub.s["_capex_hist"], acc.s["_capex_hist"][idx])
    both = AC.Accounts.concat([sub, acc.take([1, 2])])
    assert both.n == 6 and np.array_equal(both.s["dhp"][-2:], acc.s["dhp"][[1, 2]])
    x, y = stub_draws(6, 1, seed=9)[0]
    assert both.step(x, y, {})["ok"]                 # pooled books keep their identities
    cp = acc.copy()
    cp.s["tga"][0] += 1
    assert cp.s["tga"][0] != acc.s["tga"][0]


# ---------------------------------------------------------------- I9 views and account facts

def test_views_equal_the_account_ratios_and_follow_the_v3_interest_burden_definition():
    acc, _, _ = run_path(paths()["A"])
    v = acc.views()
    S = acc.s
    assert np.allclose(v["s_debt"], S["dhp"] / S["gdp"] * 100)
    assert np.allclose(v["s_foreign_hold"], S["fh"] / S["dhp"] * 100)
    assert np.allclose(v["s_int_burden"], S["_y10_win"].mean(axis=1) * v["s_debt"] / 100)
    assert S["_y10_win"].shape == (N, 20)
    assert np.all((v["s_int_burden"] > 0) & (v["s_int_burden"] < 15))
    assert acc.resid["I9_views"] < 1e-9
    assert set(acc.facts()) == {"debt_limit_binding", "xdate_passed", "arrears"}
    assert set(acc.fact_shares()) >= {"debt_limit_binding", "xdate_passed", "arrears", "hyperscaler_fcf_negative"}


def test_the_accounts_never_hand_the_model_a_learned_input():
    acc = AC.Accounts.from_root(synthetic_root(), 5)
    assert set(acc.views()) == set(AC.VIEW_FIELDS)
    assert "s_payout" not in acc.views()                 # an identity state the model's update_state owns


# ---------------------------------------------------------------- T-L7 gates

def test_debt_limit_gate_fires_at_the_root_and_closes_after_a_raise():
    _, _, gA = run_path(paths()["A"])
    _, _, gB = run_path(paths()["B"])
    assert gA[0]["congress"]["p_limit_binds_next_q"] >= 0.5
    assert gA[0]["congress"]["require_options"] == ["debt_limit_raise", "debt_limit_suspend", "debt_limit_standoff"]
    assert gA[2]["congress"]["p_limit_binds_next_q"] < 0.5 and gA[2]["congress"]["require_options"] == []
    assert gB[2]["congress"]["p_xdate_next_q"] > gA[2]["congress"]["p_xdate_next_q"]
    assert max(g["congress"]["p_xdate_next_q"] for g in gB) >= 0.5
    assert any(g["synth"]["xdate_mechanism_allowed"] for g in gB)


def test_a_debt_limit_suspension_lifts_the_constraint_and_resets_the_limit_after():
    acts = [A.account_actions([A.option_payload("congress", "debt_limit_suspend", 2)])] + [{}] * 3
    acc, reports, g = run_path(acts)
    assert all(r["ok"] for r in reports)
    assert g[0]["congress"]["p_limit_binds_next_q"] == 0.0
    assert np.all(acc.s["susp_q"] == 0)


def test_hyperscaler_buyback_tilt_bites_while_fcf_is_negative_not_after_a_capex_cut():
    _, _, gA = run_path(paths()["A"])
    _, _, gC = run_path(paths()["C"])
    assert gA[4]["bigtech"]["tilt"]["buyback_up"] == 0.5
    assert gC[4]["bigtech"]["tilt"]["buyback_up"] == 1.0
    assert gA[0]["bigtech"]["debt_needed_if_raise_15pct_4q"] > 0


def test_china_reserve_gate_and_the_saving_rate_flag():
    acc = AC.Accounts.from_root(synthetic_root(), 10)
    for x, y in stub_draws(10, 4, d_usd=12.0):        # a strong dollar four quarters running
        acc.step(x, y, {})
    g = AC.gates(acc)
    assert g["china"]["reserves_change_4q_pct"] < -10 and g["china"]["tilt"] == {"cny_weaker": 0.5}
    assert isinstance(g["synth"]["saving_rate_below_p1"], bool)


def test_hyperscaler_capex_guidance_runs_four_quarters_then_stops():
    acts = [{}] + [A.account_actions([A.option_payload("bigtech", "capex_raise_large", 20)])] + [{}] * 6
    acc, _, _ = run_path(acts, n=10)
    cap = acc.s["bt_capex"]
    assert np.allclose(cap, 230.0 * 1.05 ** 4)


# ---------------------------------------------------------------- packets: T-L4 privacy, T-L5 determinism, T-L8 budget

ACTOR_IDS = ("fed", "congress", "treasury", "president", "china", "opec", "bigtech", "voter", "courts", "belligerents")


@pytest.mark.parametrize("actor", ACTOR_IDS)
def test_every_own_block_fits_the_budget_at_the_root_and_on_a_branch(actor):
    root = synthetic_root()
    acc, _, _ = run_path(paths()["A"][:2])
    for block in (AC.own_block(actor, root, None, institutions={}), AC.own_block(actor, root, acc, institutions={})):
        assert block.startswith(f"## 你的账户：{AC.ZH[actor]}")
        assert len(block) <= AC.OWN_BLOCK_MAX, (actor, len(block), block)
        assert "nan" not in block


def test_private_notes_reach_only_their_owner_in_a_separate_call():
    sigs = [AC.Signal("bigtech", "2027 年资本开支继续增长", None, 1, 4, private_note="若纳指跌超20%则削减"),
            AC.Signal("fed", "未来决定取决于数据", "d_ff", 0, 1, private_note="倾向于12月降息")]
    root = synthetic_root()
    board = AC.signal_board(sigs, {}, viewer="president")
    assert all(s.private_note not in board for s in sigs) and "可信度 0.50（0/0）" in board
    pk = AC.root_packets(root, signals=sigs, institutions={}, show_private_for="fed")
    assert "倾向于12月降息" in pk["fed"]["text"]
    assert all("倾向于12月降息" not in p["text"] for a, p in pk.items() if a != "fed")
    assert all("若纳指跌超20%则削减" not in p["text"] for p in pk.values())
    merged = AC.root_packets(root, signals=sigs, institutions={})
    assert all("仅你可见" not in p["text"] for p in merged.values())
    assert "美联储［表态" not in AC.signal_board(sigs, {}, viewer="fed")       # an actor's own signal is not shown to it


def test_opec_spare_capacity_is_private_to_opec():
    root = synthetic_root()
    inst = {"opec_spare_mbd": 1.5}
    assert "闲置产能 1.50" in AC.own_block("opec", root, None, institutions=inst, show_private=True)
    assert "闲置产能 1.50" not in AC.own_block("opec", root, None, institutions=inst, show_private=False)


def test_rendering_is_deterministic_and_insensitive_to_noise_below_display_precision():
    root = synthetic_root()
    a = AC.root_packets(root, institutions={})
    b = AC.root_packets(root, institutions={})
    assert all(a[k]["text"] == b[k]["text"] and a[k]["hash"] == b[k]["hash"] for k in a)
    assert AC.sig3(2893.77) == AC.sig3(2893.77 * (1 + 1e-6)) == "2,894"
    acc1, _, _ = run_path(paths()["A"][:2], seed=3)
    acc2, _, _ = run_path(paths()["A"][:2], seed=3)
    assert AC.own_block("congress", root, acc1) == AC.own_block("congress", root, acc2)


def test_merged_call_increment_stays_within_the_token_budget():
    root = synthetic_root()
    pk = AC.root_packets(root, institutions={})
    own = sum(p["own_tokens_est"] for p in pk.values())
    dig = AC.est_tokens(AC.news_digest(AC.root_digest_items(root, {})))
    assert own + dig <= 2500


# ---------------------------------------------------------------- T-L9 credibility, T-L10 the news guard

def test_credibility_is_beta_2_2_with_excused_signals_left_out():
    c = AC.Credibility()
    c.update(True)
    c.update(True)
    c.update(False)
    c.update(False, excused=True)
    assert abs(c.mean - 4 / 7) < 1e-12 and c.total == 3 and c.label() == "可信度 0.57（2/3）"
    assert AC.settle_signal(AC.Signal("fed", "x", direction=-1), -1) == (True, False)
    assert AC.settle_signal({"direction": 1}, -1) == (False, False)
    assert AC.settle_signal({"direction": 1}, -1, condition_fired=True) == (False, True)


def test_the_news_digest_drops_price_forecasts_and_keeps_facts():
    kept, dropped = AC.digest_guard([("data", "预计油价将涨至 130 美元"), ("ledger", "联邦债务距法定上限 713 十亿美元"),
                                     ("data", "美股收益率会跌至低位")])
    assert dropped == ["预计油价将涨至 130 美元", "美股收益率会跌至低位"] and len(kept) == 1
    root = synthetic_root()
    items = AC.root_digest_items(root, {"fomc_2026_10": "hold"})
    assert AC.digest_guard(items)[1] == []
    text = AC.news_digest(items + [("event", "一" * 80)] * 3)
    lines = text.splitlines()[1:]
    assert len(lines) <= 6 and all(len(x) <= 52 for x in lines)
    assert lines[0].startswith("- 一")                      # events come first


def test_sector_summary_and_regime_line_are_short_and_flag_constraints():
    acc, _, _ = run_path(paths()["B"][:3])
    s = AC.sector_summary(acc)
    assert s.startswith("部门账户摘要：") and len(s) <= 250
    assert "上限约束" in s
    line = AC.regime_line({"s_core_yoy": 4.5}, {"oil_disruption_ongoing": True}, "2026Q4", acc)
    assert "高通胀" in line and "石油供给中断：持续" in line and "选举季：11 月 3 日中期选举" in line


def test_node_detail_lists_every_book_with_medians_bands_identities_and_gates():
    acc, _, _ = run_path(paths()["A"][:2])
    d = acc.detail(synthetic_root())
    assert set(d["books"]) == set(AC.BOOKS)
    fed = {ln["id"]: ln for ln in d["books"]["fed"]["lines"]}
    assert fed["reserves"]["p10"] <= fed["reserves"]["median"] <= fed["reserves"]["p90"]
    assert fed["reserves"]["update_zh"] == "恒等式" and fed["currency"]["calibration"] == "A10"
    assert all(v["ok"] for v in d["identities"].values())
    assert {"congress", "fed", "bigtech", "china", "opec", "synth"} <= set(d["gates"])
    json.dumps(AC.jsonio.clean(d), ensure_ascii=False)


def test_from_root_names_the_missing_lines():
    root = synthetic_root()
    del root["books"]["fed"]["lines"]["tga"]
    with pytest.raises(KeyError, match="fed.tga"):
        AC.Accounts.from_root(root, 3)


def test_institution_pins_override_the_judgment_parameters():
    p = AC.Params.from_root(synthetic_root(), {"em_capacity_bn": 500.0, "tga_target_bn": 700.0})
    assert p.em_capacity == 500.0 and p.tga_target == 700.0
    assert abs(p.state_local_nl_saar - (-450.0)) < 1e-9 and p.rmp_hold_q == 57.0


# ---------------------------------------------------------------- the build step's downloads

class FakeResp:
    def __init__(self, code=200, body=b"{}"):
        self.status_code = code
        self.content = body


class FakeSession:
    def __init__(self, answers=None):
        self.calls = []
        self.answers = answers or {}

    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
        self.calls.append({"url": url, "params": params, "headers": dict(headers or {}),
                           "allow_redirects": allow_redirects})
        ans = self.answers.get(url.split("?")[0], FakeResp())
        if isinstance(ans, Exception):
            raise ans
        return ans


def test_sec_requests_need_the_user_agent_and_go_only_to_sec_hosts(tmp_path):
    sess = FakeSession()
    f = AC.Fetcher(tmp_path, session=sess, sec_user_agent="", sleep=lambda s: None)
    assert f.sec("MSFT", "Revenues") is None
    assert sess.calls == [] and f.status["sec:MSFT:Revenues"]["why"] == AC.NO_SEC_UA
    cached = tmp_path / "sec" / "MSFT_Revenues.json"
    cached.parent.mkdir(parents=True)
    cached.write_text('{"units": {"USD": []}}', encoding="utf-8")
    assert f.sec("MSFT", "Revenues") == {"units": {"USD": []}}               # the cache is used without the agent
    ua = "Example Researcher research@example.org"
    f2 = AC.Fetcher(tmp_path / "c2", session=sess, sec_user_agent=ua, sleep=lambda s: None)
    f2.sec("AMZN", "Revenues")
    call = sess.calls[-1]
    assert AC.is_sec_host(call["url"]) and call["url"].startswith("https://data.sec.gov/")
    assert call["headers"]["User-Agent"] == ua
    assert call["allow_redirects"] is False                                  # a redirect never carries the agent away
    assert ua not in json.dumps(f2.status)                                   # never recorded
    redirect = FakeSession({"https://data.sec.gov/api/xbrl/companyconcept/CIK0001018724/us-gaap/Assets.json":
                            FakeResp(301, b"")})
    f3 = AC.Fetcher(tmp_path / "c3", session=redirect, sec_user_agent=ua, sleep=lambda s: None)
    assert f3.sec("AMZN", "Assets") is None and all(c["allow_redirects"] is False for c in redirect.calls)
    from scenariolab.llm import claude_code as cc
    assert "SEC_USER_AGENT" not in cc.child_env({"SEC_USER_AGENT": ua, "PATH": "/bin", "HOME": "/h"})
    assert not AC.is_sec_host("https://data.sec.gov.example.com/x") and not AC.is_sec_host("http://www.sec.gov/x")
    assert AC.is_sec_host("https://www.sec.gov/x") and AC.is_sec_host("https://sec.gov/x")


def test_a_failed_download_falls_back_to_the_cache_and_is_labelled_stale(tmp_path):
    rel = "nyfed/soma_summary.json"
    url = AC.JSON_SOURCES[rel][0]
    sess = FakeSession({url: FakeResp(503)})
    f = AC.Fetcher(tmp_path, refresh=True, session=sess, sleep=lambda s: None)
    assert f.json_source(rel) is None and f.status[rel]["status"] == "missing"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_text('{"soma": {"summary": []}}', encoding="utf-8")
    assert f.json_source(rel) == {"soma": {"summary": []}}
    assert f.status[rel] == {"status": "stale", "file": rel, "why": AC.STALE_DOWNLOAD}
    good = FakeSession({url: FakeResp(200, b'{"soma": {"summary": [1]}}')})
    f3 = AC.Fetcher(tmp_path, refresh=True, session=good, sleep=lambda s: None)
    assert f3.json_source(rel) == {"soma": {"summary": [1]}} and f3.status[rel]["status"] == "downloaded"
    bad = FakeSession({url: FakeResp(200, b"<html>moved</html>")})
    f4 = AC.Fetcher(tmp_path, refresh=True, session=bad, sleep=lambda s: None)
    assert f4.json_source(rel) == {"soma": {"summary": [1]}} and f4.status[rel]["status"] == "stale"


def test_offline_builds_never_download(tmp_path):
    sess = FakeSession()
    f = AC.Fetcher(tmp_path, download=False, session=sess, sleep=lambda s: None)
    assert f.json_source("fiscal/mspd_table_1.json") is None and sess.calls == []
    with pytest.raises(FileNotFoundError):
        f.fred("ZZZNOTASERIES")


def test_fred_reads_the_research_window_when_the_shared_cache_lacks_the_series(tmp_path, monkeypatch):
    monkeypatch.setattr(AC.config, "RAW_DIR", tmp_path / "raw")
    agents = tmp_path / "agents"
    (agents / "fred").mkdir(parents=True)
    (agents / "fred" / "ZZTEST.csv").write_text("observation_date,ZZTEST\n2026-07-01,1.5\n2026-10-01,2.5\n", encoding="utf-8")
    f = AC.Fetcher(agents, download=False)
    s = f.fred("ZZTEST")
    assert list(s.values) == [1.5, 2.5] and f.status["fred:ZZTEST"]["status"] == "cached"


def test_import_research_copies_bytes_only_and_prefers_the_long_window(tmp_path):
    dl = tmp_path / "dl"
    (dl / "fred_20261101").mkdir(parents=True)
    (dl / "fred_20261101" / "GDP.csv").write_text("observation_date,GDP\n2020-01-01,1\n", encoding="utf-8")
    (dl / "fred_20261101" / "GDP_long.csv").write_text("observation_date,GDP\n1960-01-01,1\n2020-01-01,1\n", encoding="utf-8")
    (dl / "sec_20261101").mkdir()
    (dl / "sec_20261101" / "MSFT_Revenues.json").write_text("{}", encoding="utf-8")
    (dl / "sec_20261101" / "evil.py").write_text("raise SystemExit", encoding="utf-8")
    (dl / "fed_20261101").mkdir()
    (dl / "fed_20261101" / "nyfed_soma_summary.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "cache"
    res = AC.import_research(dl, dest=dest)
    assert res["imported"] == 3
    assert (dest / "fred" / "GDP.csv").read_text(encoding="utf-8").count("\n") == 3
    assert (dest / "nyfed" / "soma_summary.json").exists() and not list(dest.rglob("*.py"))
    log = json.loads((dest / "import_log.json").read_text(encoding="utf-8"))
    assert len(log["imported"]) == 3 and all(len(r["sha256"]) == 64 for r in log["imported"])
    (dl / "sec_20261101" / "MSFT_Revenues.json").write_text('{"changed": 1}', encoding="utf-8")
    assert AC.import_research(dl, dest=dest)["skipped"] == 1                 # never overwritten without --force
    assert AC.import_research(dl, dest=dest, force=True)["skipped"] == 0


def test_a11_spells_find_flat_debt_windows_followed_by_a_jump():
    days = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    vals = np.where(days < pd.Timestamp("2023-01-19"), np.linspace(31000, 31400, len(days)), 31400.0)
    vals = np.where(days > pd.Timestamp("2023-06-02"), 31400.0 + 450.0 + (days > pd.Timestamp("2023-06-02")).cumsum(), vals)
    eps = AC.a11_spells(pd.Series(vals, index=days))
    assert len(eps) == 1 and 120 <= eps[0]["days"] <= 140 and eps[0]["start"].startswith("2023-01")


# ---------------------------------------------------------------- the build step and the server

def test_only_agents_accounts_runs_without_the_engine_context(monkeypatch, tmp_path):
    from scenariolab import build

    def no_ctx(*a, **k):
        raise AssertionError("agents_accounts must not load the engine context")

    def no_panel(refresh=False):
        raise AssertionError("agents_accounts must not build the panel")

    calls = []
    monkeypatch.setattr(build, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(build, "save_panel", no_panel)
    monkeypatch.setattr(build.config, "PANEL_PATH", tmp_path / "missing_panel.csv")
    monkeypatch.setattr(build.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(build.EngineContext, "load", classmethod(lambda cls, **kw: no_ctx()))
    monkeypatch.setattr(build, "_run", lambda module, ctx, **kw: calls.append((module, ctx, kw)) or {})
    assert build.main(["--only", "agents_accounts"]) == 0
    assert calls == [("agents.accounts", None, {"refresh": False})]
    assert "agents_accounts" in build.STEPS and "agents_accounts" not in build.OPT_IN


def test_the_server_serves_the_root_and_names_its_build_step(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from scenariolab import server
    monkeypatch.setattr(server, "RESULTS", tmp_path)
    c = TestClient(server.app, raise_server_exceptions=False)
    r = c.get("/api/agents_accounts_root")
    assert r.status_code == 404 and "--only agents_accounts" in r.json()["detail"]
    (tmp_path / "agents_accounts_root.json").write_text('{"books": {}}', encoding="utf-8")
    assert c.get("/api/agents_accounts_root").json() == {"books": {}}
    assert server.BUILD_STEP["agents_priors"] == "agents_accounts"


def test_run_writes_the_root_and_the_priors_and_refuses_an_empty_build(monkeypatch, tmp_path):
    from scenariolab.agents import priors
    monkeypatch.setattr(A, "INSTITUTIONS_PATH", tmp_path / "agents" / "institutions.json")
    monkeypatch.setattr(priors, "build_priors", lambda data=None: {"version": priors.PRIORS_VERSION, "summary": {}})
    monkeypatch.setattr(AC, "build_root", lambda fetch=None: {"books": {"fed": {"lines": {"tga": _line(1.0)}}},
                                                              "stale": [], "missing": [], "source_summary": {},
                                                              "checks": {}})
    out = AC.run(results_dir=tmp_path, download=False)
    assert out["lines"] == 1 and (tmp_path / "agents_accounts_root.json").exists()
    assert (tmp_path / "agents_priors.json").exists() and (tmp_path / "agents" / "institutions.json").exists()
    monkeypatch.setattr(AC, "build_root", lambda fetch=None: {"books": {}, "stale": [], "missing": [],
                                                              "source_summary": {}, "checks": {}})
    with pytest.raises(RuntimeError, match="账户数据全部缺失"):
        AC.run(results_dir=tmp_path, download=False)


def test_build_root_records_missing_sections_instead_of_failing(tmp_path, monkeypatch):
    monkeypatch.setattr(AC.config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(AC.config, "PANEL_PATH", tmp_path / "no_panel.csv")
    root = AC.build_root(AC.Fetcher(tmp_path / "agents", download=False, sec_user_agent=""))
    assert root["books"] == {} and {m["book"] for m in root["missing"]} >= {"fed", "fiscal", "bigtech"}
    assert root["sources"] and all(v["status"] == "missing" for v in root["sources"].values())


# ---------------------------------------------------------------- the real cache (skipped on a fresh clone)

needs_cache = pytest.mark.skipif(not (config.RAW_DIR / "agents" / "fiscal" / "dts_debt_subject_to_limit.json").exists(),
                                 reason="agents cache not imported (python -m scenariolab.agents.accounts --import ...)")


@needs_cache
def test_the_cached_root_reproduces_the_research_snapshot():
    root = AC.build_root(AC.Fetcher(download=False))
    v = lambda b, k: AC.root_value(root, b, k)   # noqa: E731
    if v("fiscal", "statutory_limit") != 41103.996:
        pytest.skip("the cache holds another vintage than the research round's (2026-10-29)")
    assert abs(v("fiscal", "limit_headroom") - 713.22) < 0.01
    assert abs(v("fed", "total_assets") - 6745.635) < 0.01 and abs(v("fed", "deferred_asset") - 232.521) < 0.01
    assert abs(v("bigtech", "capex") - 230.422) < 0.01 and abs(v("bigtech", "fcf") + 35.329) < 0.01
    assert root["checks"]["A11"]["n_since_2011"] == 10 and root["checks"]["A12"]["reproduced"]
    acc = AC.Accounts.from_root(root, 200)
    for x, y in stub_draws(200, 2):
        assert acc.step(x, y, {})["ok"]


# ---------------------------------------------------------------- watched lines, account events, refunds

def test_own_blocks_show_at_most_two_watched_lines_from_other_books():
    root = synthetic_root()
    fed = AC.own_block("fed", root, None, institutions={})
    assert "关注（他人账户）：联邦上限余量 713｜财政部 TGA 1,040" in fed
    assert "选民人均实际可支配收入" in AC.own_block("congress", root, None, institutions={})
    assert "美国战略石油储备" in AC.own_block("opec", root, None, institutions={})
    assert "关注" not in AC.own_block("voter", root, None, institutions={})


def test_account_events_report_thresholds_crossed_this_quarter():
    acc = AC.Accounts.from_root(synthetic_root(), 50)
    seen = []
    for x, y in stub_draws(50, 2):                  # headroom 713: the limit binds in the second quarter (2027Q1)
        before = acc.fact_shares()
        acc.step(x, y, {})
        seen.append(AC.account_events(before, acc))
    assert not any("触及法定上限" in t for _, t in seen[0])
    ev = seen[1]
    assert any("触及法定上限" in t for _, t in ev) and all(k == "ledger" for k, _ in ev)
    assert AC.account_events(acc.fact_shares(), acc) == []              # nothing new once it has happened
    assert AC.digest_guard(ev)[1] == []


def test_a_refund_ruling_pays_the_pinned_liability_once():
    p = AC.Params.from_root(synthetic_root(), {"refunds_remaining_bn": 120.0})
    acc = AC.Accounts.from_root(synthetic_root(), 20, params=p)
    acts = A.account_actions([A.option_payload("courts", "strike_down_refunds")])
    assert acts == {"customs_refunds_bn": "pinned"}
    x, y = stub_draws(20, 1)[0]
    rep = acc.step(x, y, acts)
    assert rep["ok"] and np.allclose(acc.flows["refunds_q"], 120.0)
    x, y = stub_draws(20, 1, seed=4)[0]
    acc.step(x, y, {})
    assert np.allclose(acc.flows["refunds_q"], 0.0)


def test_root_gates_read_the_a2_cash_deficit_before_any_step():
    acc = AC.Accounts.from_root(synthetic_root(), 20)
    g = AC.gates(acc)
    assert np.allclose(acc.flows["mts_def_q"], -13.2 + 1.055 * 0.052 * 33000.0 / 4)
    assert g["congress"]["p_limit_binds_next_q"] == 0.0          # headroom 713 > one quarter's need: binds in 2027Q1
    assert g["bigtech"]["tilt"]["buyback_up"] == 0.5             # FCF < 0 at the root (G-L7)
    assert g["fed"]["reserves"] < 2900 and g["fed"]["tilt"] == {"bs_runoff": 0.5}   # G-L5 at 2.89tn


def test_a_sec_404_is_cached_so_the_next_build_sends_nothing(tmp_path):
    url = "https://data.sec.gov/api/xbrl/companyconcept/CIK0001326801/us-gaap/RevenueRemainingPerformanceObligation.json"
    sess = FakeSession({url: FakeResp(404)})
    f = AC.Fetcher(tmp_path, session=sess, sec_user_agent="Example research@example.org", sleep=lambda s: None)
    assert f.sec("META", "RevenueRemainingPerformanceObligation") is None and len(sess.calls) == 1
    f2 = AC.Fetcher(tmp_path, session=sess, sec_user_agent="Example research@example.org", sleep=lambda s: None)
    assert f2.sec("META", "RevenueRemainingPerformanceObligation") is None and len(sess.calls) == 1
    assert f2.status["sec:META:RevenueRemainingPerformanceObligation"]["status"] == "absent"
