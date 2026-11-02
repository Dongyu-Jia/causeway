# SIM_DESIGN：季度情景模拟（情景 → 因子 → 资产）— implementation spec

Role: designer (label `design`). This was a read-only design round. Nothing was written in MAIN, RWT, the former FWT or
any other worktree; no LLM was called and nothing was downloaded. Date 2026-11-02. This spec reconciles the nine research
reports of the round and the critique, and fixes the defaults. It is the document an implementer builds from; the
research folders remain the detailed references it cites.

Scripts of this role are in `sim_redesign/design/` (d01–d05) with outputs in `design/work/`. Each ran as
`cd MAIN && MAIN/.venv/bin/python -I -B <script> <inputs>` and reads only scratch files. Numbers computed here carry the
tag `[d0x]`; numbers from the reports carry the report name.

| Source (all under `scratchpad/sim_redesign/` unless noted) | What this spec takes from it |
|---|---|
| `critic/critique.md` | every verdict; the ten findings and the seven design consequences (all applied, Appendix C) |
| `harness/simharness.py` | the oracle-macro evaluation harness (ported into the app, §3.9) |
| `layers/` (l01–l05, model card) | timing classes, anticipation evidence, endogeneity table, forecast-tab proposal |
| `writer/`, `narratives/E01–E12/` | episodes, narratives n1/n2/n3, the blind prompt builder |
| `valuation/` (READY.md, valuation_quarterly.csv) | valuation series with timing; the negative results on pulls and level states |
| `joint/` (jq.py, jq2–4, j11, j13) | the quarterly BVAR, Waggoner–Zha conditioning, the C′ hybrid, persistence diagnostics |
| `schema/SCHEMA.md`, `schema.json` (schema-revise-2 3.0) | the 94-field input table, policy split, templates, oracles |
| `qmodel/` (qcore.py, qplugin.py, README.md) | the quarterly three-input model G, the NN comparison, masking results, the plug-in |
| `translator/`, `narratives/*/q_n*.json` | the 36 stand-in translations, the QTable shape, the protocol slips |
| `scorer/` (out/*.json, *.csv) | the end-to-end test results and decomposition |
| `scratchpad/regime/DESIGN.md`, `scratchpad/factors/FACTORS_DESIGN.md` | the two rounds that have since landed |
| `scratchpad/agent_design/AGENT_DESIGN.md`, `scratchpad/instruments/INSTRUMENTS_DESIGN.md` | the two rounds now running, which consume this model |
| `scratchpad/USER_DECISIONS.md` | user-stated decisions (LLM backend, tests never spend tokens, snapshot mode) |
| code on `feat/causeway-v2` @ `773dfb7` (read with `git show`) | the modules this spec changes |

**Repository state when this spec was written (2026-11-02, about 01:20).** The task text describes the regime and factor
rounds as running. They have since landed, and two newer rounds depend on this spec:

| Branch / worktree | State |
|---|---|
| `main` @ `eb2f448` (MAIN) | the original app; no round has committed to it (its `.git` metadata and `.env` were changed by other processes, as sibling reports noted) |
| `feat/causeway-v2` @ `773dfb7` (integration; worktree `causeway-regime-layer` = RWT) | regime layer `9146ca4` (11-01 17:32) + Claude Code backend `db5f56f` (18:22, merged `0617f2f`) + factor round `12b4141` (19:52), merged at 22:45. RWT carries **uncommitted** regime follow-up work (`scenarios/regime_views.py`, `calibrate.py`, ten spec YAMLs, `_scenario_detail.js`, tests, README, ARCHITECTURE; last write 01:22) |
| `feat/factors` | landed; the FWT worktree `causeway-factors` no longer exists |
| `feat/agent-tree` (worktree `causeway-agent-tree`, from `773dfb7`) | running: `scenariolab/agents/` with a `ModelAdapter` protocol and a `GaussianStub`; its phase P14 swaps in "the real quarterly model" (this spec) |
| `feat/instruments` (worktree `causeway-instruments`, from `773dfb7`) | running: `scenariolab/instruments/` with `CoreView` (`freq` "M" or "Q"); it lists requirements on the quarterly model (income split, T18 correlations) |

Section 12 sets the landing order for all of them.

---

## 0. 一页总结（给用户）

**你要的东西，按本设计的读法。** 预测只在第一层：带概率的情景假说。第二层用一个内部一致的联合模型，把情景没说的因子全部补齐。第三层把因子映射成资产，可以是同期的，但每个输入都标明是“同期”“前定（季初已知）”还是“假设（情景给的）”。只解释季度和更长的期限。当前估值既是状态，也是要建模的一条腿。模型按季度运行：输入是季初以前的全部数据 s_q、季前情景文字 c_q 和季内情景文字 x_q，大模型只翻译两段文字，输出季末 13 个资产的联合分布；同一个模型一季一季往前滚，推 10–15 年。事件一律拆成通用渠道，没有单一事件的开关。

**研究轮查到、并经批评复核的六件事：**

1. **宏观本身对资产几乎没有信息。** 把真实的工业产出、商品与服务通胀、政策利率和失业率原样喂给传导模型，1985–2026 年合并的样本外 R²（10 个资产平均）在 1/4/12/20 个季度是 −0.08 / −0.15 / −0.37 / −0.42（独立窗口 167 / 41 / 13 / 8 个）。“已知宏观”的能力全部来自信用利差：只用 GZ 信用利差就有 +0.17 / +0.15 / +0.11 / +0.11，因子轮的完整模型 F 是 +0.23 / +0.22 / −0.01 / −0.03。
2. **决定资产结果的是情景对金融条件的陈述。** 所以在“情景推因子、因子推资产”里，起作用的是情景对信用利差和风险溢价怎么说。说对它几乎等于说对了股权风险溢价，这正是“预测完全交给情景假说层”的含义，界面会明说。本设计把金融条件改为每个阶段必填。
3. **正确的叙事没有带来可检出的改进，而且检验不是盲的。** 在 11 个历史片段里，能检出的最小差异约 0.02–0.03 log，叙事的改进小于它。翻译者 12 个片段全部认出了年代；不含数字的 n3 也几乎复原了真实的信用利差路径（2020 年那个片段相关系数 0.99）。精确的因子路径在 8 个普通片段里有用（MSE 技能 +0.27，事后挑出的子集），在 2007–09 和 2001–02 两次大跌里完全失灵：倍数崩了，而利率、宏观和历史漂移都指向上涨。
4. **起点估值不能当预测量用。** 起点估值（CAPE、P/D、自由现金流收益率、超额 CAPE 收益率）对 1–20 个季度的回报没有样本外增益，任何“拉回均值”的设定都更差。在 10 年尺度上 CAPE 能给各个十年排序，但 1980 年以后水平不准。所以估值是状态和初始条件，用来算收入、做拆分和显示。情景可以明说“估值重估”，但那等于直接假设股价，界面会这样写；模型不会偷偷回归。“股价 = 基本面 × 倍数”的拆分改用 CAPE 倍数：滚动 P/E 被减记扭曲，2007–09 年股价跌了 0.47，P/E 的“倍数”反而涨了 1.0。
5. **你问的神经网络做了，但输了。** 它的输入是 [s_q, x_q⊙m, m]，m 标明文字给了哪些字段。在三个十年里它都输给结构化的线性高斯模型：1 季 R² 低 0.21、0.22、0.31，90% 区间都不含 0。第一版用线性高斯模型；神经网络只作对照，只有在预注册检验里每个时段都赢才会启用。
6. **现在的月度引擎有两个缺陷。** 它只交付了叙事里宏观路径的 40–70%，原因是熵池收缩。回测里的长期锚（r* 0.75、期限溢价 0.75、利差 2.0、失业率 4.8）用的是 2008 年以后才知道的水平。

**改成什么：**

- **一个季度联合模型：p(因子 | 季初状态) × p(资产 | 因子, 季初状态)。**
  - 因子层是贝叶斯 VAR：10 个核心变量，其余字段作卫星方程。
  - 文字给的值精确满足（Waggoner–Zha 条件模拟），没给的从联合分布补全。
  - 政策可以写成“干预”（do 算子）：同季的宏观不随干预补全，市场价格会随之变化。
  - 零下限、长期锚和经济周期的形状单独处理。
  - 资产层默认用因子轮的 F：月度估计的 β，加总到季度。其他字段和状态要通过预注册检验才进入。
- **大模型的输出格式就是模型的输入格式。** 这个格式叫季度情景表 QTable，由同一张变量卡片生成。每个值都带来源（文字里的数字、文字里的定性幅度、推算）和原文引文。文字没给的就留空，由模型补全，并在界面上标出来。
- **叙事检验成为应用里的常设功能：**
  - 18 个历史片段，新增 1987–88、1998–2000、2010–12、2024–26，另有两个训练期较短的片段单独报告；
  - 盲测规范、错叙事对照和合成片段；
  - 一个不用大模型的事前检验：“美联储当时的预测（SEP）”。

  真实大模型只在你点按钮或运行命令、看过调用次数估算并确认之后才调用。所有测试和构建都用模拟后端，不花额度。
- **页面改动。** 一年期页面改为“未来一年：情景条件模拟”。新的“情景条件检验”页按季度和年度报告，取代月度 R² 的标题。
- **过渡安排。** 季度模型通过切换关（§4.6）之前，13 个内置情景、组合评估和提取页继续用月度引擎。季度模型先给情景实验室和智能体推演用，标“试运行”。

**需要你定的事**（默认值见 §13）：

- 切换默认引擎的条件；
- 金融条件是否必填；
- 估值默认模式和倍数口径；
- 长期回报是否扣除历史估值抬升（每年约 1.2–1.9%）；
- 真实调用用本机 Claude Code 还是 API；
- 新增哪些检验片段。

## 0.1 Decisions on one page

| # | Decision | Default in this spec | Why / which test decides (§) |
|---|---|---|---|
| D1 | Core architecture: a quarterly cell with inputs s_q, c_q, x_q and one joint model p(x \| s) · p(r \| x, s) | adopted | the user's 4th message; the unstructured joint Gaussian lost badly (−0.11 / −0.19 / −0.73), the structured factorization did not (§3) |
| D2 | Asset layer default = L0: the factor round's F, with monthly-fitted betas applied to quarterly sums | adopted | G's margin over L0 is the size of a selection effect (critic 7); quarterly-only fits lose (§4.1) |
| D3 | Extra asset-layer terms (tier-1 fields beyond F, states, interactions, next-quarter leads) | candidates only | pre-registered test T-A, selection on 1990–2014 and a single score on 2015–26 (§4.2) |
| D4 | Neural network | report-only | it lost in every split: NN − G = −0.21 / −0.22 / −0.31, all intervals below 0 (§4.3) |
| D5 | Factor layer = BVAR core + satellites; exact conditioning; do-operator; ZLB shadow rate; cycle module; anchors knowable at the train end | adopted for the quarterly stack | acceptance tests T-F and T-C (§5) |
| D6 | Financial conditions (credit stance) required in every phase | required; if missing, the model fills it in and the UI says it has no predictive value | credit spreads carry all the oracle skill (critic 1) (§7.1) |
| D7 | Valuation: an identity on the CAPE multiple, no pull; scenario modes inferred / hold / rerate with sizes taken from historical quantiles; reversion only when stated, half-life ≥ 14 years | adopted | every out-of-sample pull lost; P/Ê is distorted by write-downs (§6) |
| D8 | Forward equity drift excludes the historical CAPE re-rating (1.2–1.9 %/yr) | on (user decision U-5) | extending a 53-year re-rating is an unstated valuation forecast (§4.4) |
| D9 | LLM output = QTable, generated from the model card; every value carries a quote and a source | adopted | "the LLM's output schema must equal the model's input schema" (§3.4, App. B) |
| D10 | Narrative test is a permanent app feature: 18 episodes, blind protocol, oracle / narrative / wrong-narrative / synthetic / SEP arms; mock backend in tests; real runs only by hand | adopted | the user's headline test; the research version was not blind (§8) |
| D11 | The monthly engine stays the default until the quarterly stack passes the switch gate T-S | adopted | the research qmodel lost to the engine on narratives by +0.056 to +0.067 log (§4.6) |
| D12 | Forecast tab becomes 未来一年：情景条件模拟 | adopted | prediction lives only in the scenario layer (§9) |
| D13 | New evaluation tab 情景条件检验 replaces the monthly-R² headlines | adopted | the user asked for quarterly and longer horizons (§10) |
| D14 | Fixes to today's app first (P0): train-end anchors in backtests, fan_cond split, CAPE data, x_oil_cause, translator refusals | adopted | look-ahead and data errors confirmed by the critic (§14) |
| D15 | Landing order: regime follow-up, then P0, then instruments and agent tree, then this spec's model phases, then its UI phases | adopted | §12 |

---

## 1. What the user asked, and the reading this spec implements

The user's messages, in order:

1. 「我们不是做"资产价格预测"……情景推因子，因子推资产……"预测则完全delegate 给 情景假说"这一层……解释季度？」
2. 「要把当前估值，也当作一个因子」 (SPY/QQQ P/E and FCF yield).
3. 「去预测2022-2023股价……情景是已知的……看模型能不能预测」
4. The quarterly model with three inputs: the data before the quarter, the pre-quarter text and the in-quarter text. The LLM translates the two texts; the model predicts the end-of-quarter prices. The user asks what the model's factor inputs are and how they become neural-network inputs.
5. 「pandemic 太特殊了，本质是什么」: no single-episode flags; decompose events into generic channels.

| User statement | Reading implemented here | § |
|---|---|---|
| 情景 → 因子 → 资产; prediction is delegated to the scenario hypotheses | Layer 1 (hypotheses with probabilities) is the only predictive layer; Layers 2 and 3 simulate | 3.1 |
| Same-period inputs are fine if labelled | every input carries 同期 / 前定 / 假设 and a timing class in the model card | 3.3, App. A |
| One model, internally consistent | one joint model; stated values are delivered exactly, the rest come from the joint conditional distribution | 3.2, 5 |
| Quarterly, not monthly | quarterly cell; evaluation at 1Q and 4Q; 12Q and 20Q shown only as anecdotes | 3, 10 |
| Valuation is a factor and a state | valuation states, identity legs and scenario valuation modes | 6 |
| The 2022–23 end-to-end test | the permanent narrative test, with 2022–23 as E10 | 8 |
| What the factor inputs are and how they become network inputs | QTable fields = model fields; masked inputs are completed by Gaussian conditioning; the NN was tested and is report-only | 3.4, 3.5, 4.3 |
| The pandemic is generic channels | translator decomposition into templates; no flags; no dated sampling weights in the new stack | 7.2, 5.2 |

**Answer to the 4th message in one paragraph** (also shown on the method tab).
- **The factor inputs.** They are the 52 x_q fields of Appendix A, in families: monetary, fiscal, demand, supply, prices and activity, financial conditions, risk and policy. Five policy shocks u_* write interventions. c_q carries anchors, announced paths, valuation statements and regime requests. The 22 s_q states come from data.
- **What the LLM writes.** It writes a QTable: for each quarter, each field is either absent (not stated) or carries a value, a qualitative size or a range, always with a quote from the text.
- **How the three inputs enter the model.** The model reads [s_q, x_q ⊙ m, m], where m marks which fields the text stated. In the Gaussian model the mask needs no learned embedding: the stated components are conditioned on and the others are integrated out.
- **The neural network.** A masked MLP on exactly these inputs was trained and lost in every decade (§4.3). The first version is therefore the structured linear-Gaussian model.

---

## 2. The evidence after the critique, and what each finding changes

Independent units are given in brackets. "Pooled rolling" means origins 1985–2026 refitted every January. "Split" means
train ends at 1994Q4 / 2004Q4 / 2014Q4 with the next decade as the test.

| # | Finding (verdict) | Number | Source | Consequence here |
|---|---|---|---|---|
| E1 | Oracle-macro skill is credit-spread skill (CONFIRMED, critic 1) | Pooled rolling mean10 R², 1/4/12/20Q (167/41/13/7–8 windows): F +0.226/+0.223/−0.007/−0.030; F without GZ +0.128/+0.118/−0.119/−0.133; without GZ and Baa −0.003/−0.016/−0.258/−0.285; pure macro −0.081/−0.145/−0.373/−0.421; GZ alone +0.165/+0.149/+0.110/+0.108 | critic c10, c11, c18 | financial conditions are required (§7.1); arms O-A and O-B are reported separately (§8.3); the evaluation tab shows the ablation (§10.1) |
| E2 | "The misses are valuation swings" is an identity (REFUTED as evidence) | a constant-forecast placebo gives corr(miss, ΔP/D)² = 0.854 across episodes, above every model; 0.83–1.00 on the harness | critic c04, c12 | report corr(forecast, outcome) and the share of the multiple change predicted, with a placebo row (§6.6) |
| E3 | The narrative test was not blind (CONFIRMED, more channels) | 12/12 periods recognised; the number-free E08 n3 d_gz path has corr 0.99 with realized (MAE 0.20 pp against an sd of 1.16) | critic 3, translator, scorer | blind protocol and leakage controls (§8.4); recognition is measured with every run |
| E4 | "A perfect translation would not help" is overstated (PARTLY REFUTED) | without E03: E-oracle MSE skill −0.27 [−0.50, +0.28]; without E03, E05 and E12 (n 8, post hoc): +0.27 [+0.17, +0.45], while narratives stay at about 0 | critic c21–c23 | headline restated (§0); bust/ordinary strata pre-registered; more busts added (§8.2) |
| E5 | Look-ahead in every engine run (CONFIRMED) | DEFAULT_REVERSION r* 0.75, TP 0.75, spread 2.0, unemployment 4.8 in every window. Knowable at 1994Q4: real ff 2.19, term spread 1.67, spread 2.01, unemployment (80-quarter mean) 7.05 [d02, d05] | critic 5 | anchors are fitted to the train end (§5.5, P0.1) |
| E6 | Wrong-sign medians in busts are mostly drift (PARTLY REFUTED cause) | E05: intercept +0.198 vs factors +0.078; E12: +0.271 vs +0.133; the B0 unemployment beta is positive | critic c16 | drift is decomposed and shown; the re-rating is removed from forward drift (§4.4) |
| E7 | G − L0 is selection-sized (PARTLY) | G − L0 at 1Q +0.026/−0.034/+0.051, about equal to the tier-1 gain over F (+0.023/+0.003/+0.054); 16 variants were scored on the test decades | critic 7 | L0 is the default; G's additions go through T-A (§4.2) |
| E8 | Data errors (CONFIRMED) | MAIN s_cape 1.040–1.125× Shiller since 2023-07; P/Ê leg +1.005 while price fell 0.468 (E05); about 30% of the P/D rise is payout decline; x_oil_cause conventions are opposite; Shiller E +23% in 2026H1 is unexplained | critic 8 | fixes (§6.5, P0) |
| E9 | Small samples | 2–3 independent windows per decade at 12–20Q; 11 episodes; 1 of 30 paired intervals favourable and 3 unfavourable | critic 9 | n_indep printed everywhere; 12–20Q only as anecdotes; minimum detectable effect stated (§8.6, §10) |
| E10 | The scorer's audit was incomplete | 16 f_sentiment and 3 f_profit_share views had magnitudes chosen by the translator; the CTX and schema channels were missed | critic 10 | the validator requires a quote for every value; the audit is permanent (§7.3, §8.4) |
| E11 | The engine under-delivers narrated macro | 48–60% of the policy-rate move, 37–51% of unemployment, 33–40% of CPI y/y, 47–63% of oil; EP shrink in 32/52 phases (n1) | scorer | exact or soft delivery by conditioning (§5.3) |
| E12 | The quarterly G ≈ L0 at 1Q; the NN loses; the unstructured joint Gaussian loses | G +0.152/+0.226/+0.492; L0 +0.126/+0.261/+0.441; NN −0.062/+0.004/+0.185; G-full −0.111/−0.185/−0.731 (split, 40/40/47 quarters) | qmodel q05/q03, critic c13 | structured factorization; NN report-only (§4.3) |
| E13 | Quarterly OLS without the monthly prior is far worse | us_equity pooled 4/12/20Q: Fq −0.09/−0.72/−1.05 against F +0.30/−0.09/−0.11; L1-Fp (shrunk to the monthly betas) 0.137/0.255/0.439 | valuation, qmodel q03 | the asset layer borrows the monthly betas (§4.1) |
| E14 | Completing market prices from macro alone has no value | median completion R² −0.06 (pooled); d_ff −1.22 and d_y2 −1.4 to −1.9 at the ZLB; wrong sign in 2022 and 2007–09 | qmodel, scorer | completion shrinkage, regime covariance, FC required (§3.5, §5.7) |
| E15 | Saturating inputs at ±4 sd is essential for linear completion | O-A 2015–26 −0.578 without it, −0.035 with it; d_bs 2008Q4 is about 77 sd | qmodel q07 | kept (§3.5) |
| E16 | BVAR vs engine dynamics | contractions > 6Q: history 0/7, engine 18%, BVAR 55%, C′ 22%; policy-on-unemployment slope: engine −0.43 against history −1.44 [−2.02, −0.88]; 20Q coverage: BVAR 0.80/0.76 against engine 0.64/0.63 | joint | factor layer design C′ (§5) |
| E17 | No usable error correction after 1973; every pull loses | κ 0.0008/q (t 0.07); pooled 12/20Q: constant anchor −0.47/−0.75, real-yield anchor −0.25/−0.27; seven level states added to F all lower R² | valuation, critic c14 | no pull; valuation is a state (§6) |
| E18 | Policy fields are mostly reactions; QE cannot be learned | reaction-function R² 0.40–0.54 (non-ZLB, n 86–175); raw d_bs gives the wrong equity sign (−1.7%/sd) | schema v01, qmodel | do-operator; QE only through intervention paths (§5.3) |
| E19 | Endogeneity of candidate inputs | adj R² on same-quarter asset returns: P/D 0.93, oil 0.45, GZ 0.40, spread 0.32, y10 0.27, sentiment 0.24 | layers, critic c20 | E1/E2 fields only when the text states them, with a quote (§7.3) |
| E20 | The research qmodel lost to the engine on narratives | Q − E at the common end +0.056 / +0.067 / +0.064 (n1/n2/n3), every interval > 0; under oracle macro they tie (−0.017 [−0.075, +0.021]) | scorer, critic | switch gate T-S (§4.6) |
| E21 | ip and unrate are priced about a quarter ahead (PLAUSIBLE) | next-quarter leads improve quarterly R² for 9/8/7 of 10 assets; pooled bootstrap P > 0 = 0.94, interval touches 0 | layers | candidate C3 in T-A (§4.2) |
| E22 | Oil-cause split: marginal over F | +0.005/+0.006/+0.037, lower bound 0.000 | schema v02 | candidate C1 (§4.2) |
| E23 | History correlations for the joint distribution | quarterly corr(d_gz, us_equity) −0.557 (1973–2026, n 214), −0.582 (1985–2026, n 167); corr(d_spread, us_equity) −0.401 / −0.599 [d01] | d01 | T18 restated as a posterior predictive check (§11.2) |

---

## 3. Target architecture

### 3.1 The three layers

```
 Layer 1  情景假说（带概率）                     the only predictive layer
          written by: the user, the 13 built-in scenarios, the LLM translating a narrative, the agent tree
          ── QTable: per quarter, stated fields (value / size / range, each with a quote) + c_q
 Layer 2  因子路径：一个联合模型                 simulation, not prediction
          stated values delivered (hard = exactly, soft = within the stated sd); everything else drawn from
          p(x | s, stated); policy may be an intervention (do-operator)
          ── complete x_q: 52 fields + 5 shocks, per path and quarter
 Layer 3  资产路径：同期映射                     simulation
          r = drift + B·g(x_q, s_q) + residual; mechanical assets (crude, treasury, cash) by identity
          ── 13 asset quarterly log returns (price, income, total) + valuation legs
 State roll   s_{q+1} = Φ(s_q, x_q, r_q): levels, inflation trend, CAPE, P/D, income yields, regime flags
```

The three layers keep two errors apart. Hypothesis error is Layer 1 being wrong. Simulation error is Layers 2–3 being
wrong given a correct hypothesis. The narrative test measures the two separately (§8.6), and the UI never mixes them.

### 3.2 The quarterly cell

**Inputs.**
- **s_q, the state at the start of quarter q.** Every value is known at that time, after publication lags:
  - the 22 model states of Appendix A;
  - the core block's last two quarters (VAR lags) and the satellites' last quarter;
  - asset-derived states: realized volatility `s_vol` and own momentum (both used only by T-A candidates);
  - the flags `s_zlb` (s_ff < 0.5) and `s_core4` (core CPI y/y > 4% in the month before the quarter).
- **c_q, the pre-quarter text's translation:**
  - long-run anchors (r*, π_lr, a_tp, a_spread, a_ustar, a_oil, a_usd);
  - announced paths, which become soft constraints on future quarters;
  - constraints that span quarters (sums, means, y/y, ranges);
  - valuation statements;
  - regime requests.
- **x_q, the in-quarter text's translation:**
  - any subset of the 52 fields and the 5 shocks, each as a value with an optional sd, a qualitative size or a range;
  - a financial-conditions stance, required for every quarter.

**Model.**
- Factor layer. z_q stacks every x field in model units plus the shocks. Its distribution is
  p(z_q | s_q, z_{q−1}, z_{q−2}; θ), conditioned on the stated components (§5).
- Asset layer. r_{a,q} = α_a + β_a′ g(z_q, s_q) + ε_{a,q}. Here g maps schema fields to F's factors (§4.1), and ε_q is a
  quarterly residual vector (§4.5).
- Legs and income: y_mult, y_fund, y_pd, and each asset's income (§6).
- State roll: s_{q+1} = Φ(s_q, z_q, r_q). Φ is a set of accounting identities with no fitted asset parameters (§3.6).

**Output.** Paths of r_q (13 assets: log price, income, log total), z_q (the completed factors), the legs and the states.

**Training.** The factor layer is fitted on the measured quarterly panel; the realized values are the "perfect
translation". The asset layer's betas are fitted on monthly data and aggregated. At run time the translation feeds the
same fields. No separate text model is trained, so the text-to-field mapping is exactly the measured field's
definition, which the model card gives the translator.

### 3.3 The model card and the timing labels

The model card has one row per variable, 101 in all:
- x_q: 52 fields and 5 shocks;
- s_q: 22 states;
- c_q: 5 anchors;
- outputs: 5;
- views: 12.

Appendix A is generated from `schema.json` (schema-revise-2 3.0) and this spec's overrides by `design/d03_model_card.py`,
so the two cannot disagree.

Its columns:
- 字段, 中文名, 槽位;
- **同期/前定/假设**, the user's label;
- 时点: 同季·实现, 同季·市场或调查预期, 滞后一季, 期初, 情景路径 or 预告路径;
- 内生性: E0 exogenous macro, E0p policy, E1c commodity prices, E1 financial markets, E2 surveys and news, E3 built
  from equity prices, S, C, V, O;
- 到资产的路径: L0 asset equation, candidate, only through the factor layer, identity, anchor, or not in the model;
- 翻译器 (who may write it), 核心因子层, 金融条件, unit, first quarter, release lag, note.

What "leading" means here, after the evidence:
- **No x_q field leads asset returns.** Every x_q field is same-quarter (同期).
- **The s_q states are predetermined** (前定).
- **Tested leading information exists only for macro.** These are PLAUSIBLE results on n 36 annual origins: the term
  spread leads Δunemployment (out-of-sample +0.13), the real policy rate leads Δff (+0.17), and wage growth leads ΔCPI
  (+0.27). They may enter Layer 1 only, to tilt scenario probabilities after a pre-registered macro_skill test (§7.5).
  They never move asset paths.
- **Valuation levels lead 10-year returns in rank only.** They enter as initial conditions.

The card lives in `scenariolab/qmodel/card.json`. `tools/build_model_card.py` builds it from `schema.json` and the
overrides. It is consumed by the translator prompt, the QTable JSON schema, the validator, the UI's 变量卡片 tab and the
docs. A test asserts that card, QTable schema and quarterly panel columns agree one-to-one.

### 3.4 The contract: LLM output schema = model input schema (QTable/1)

The full JSON schema is `design/work/qtable.schema.json`, generated by `design/d04_qtable_schema.py`. Appendix B has a
summary and an example. The rules:

- **One QTable per scenario**, with one row per quarter, quarter index q = 1…H, H ≤ 60.
- **An absent field means "not stated"**, and the model fills it in. There are no nulls to emit, which keeps the
  output short.
- **Entry forms.** Each entry is one of:
  - `value` (optional `sd`; without `sd` it is a hard statement);
  - `sign` + `size`, with size ∈ {小, 温和, 明显, 大幅, 极端} = 0.25 / 0.5 / 1 / 2 / 3 training sd of the field's
    quarterly change;
  - `min` / `max` (a range).

  Every entry has `src` ∈ {stated (a number in the text), stated_qual (words mapped to a size), derived (arithmetic on
  stated values, e.g. a y/y path turned into quarterly rates)} and a `quote` of at most 120 characters. The validator
  requires the quote to be a substring of the narrative. Every value therefore has a textual basis. The translator's
  own choices (the "取历史下沿附近" sentiment views, critic 10) cannot pass.
- **`fc.stance` is required in every row**: 明显收紧 / 收紧 / 中性 / 宽松 / 明显宽松 / 未说明. It maps to a soft
  constraint on d_gz and d_spread at the size bins. 未说明 lets the model complete it, with the warning W-FC (§7.3).
- **`x_oil_cause`** is an enum: "supply" | "demand" | "mixed" (= 0.5). It replaces the 0/1 code whose conventions were
  opposite between the translator and qdata (critic 8).
- **`u.*` shocks are written only for interventions the text names**: an emergency or intermeeting cut, a regime
  change, a pre-emptive hike, QE/QT, a tax law, stimulus checks. A reaction to the economy is written as the macro,
  and the policy rate then follows from the reaction function (§5.3).
- **`c_q` contents:**
  - `anchors`, each with a quote;
  - `announced` paths (soft constraints with sd);
  - `constraints`: sum / mean / yoy / level / range over named quarters;
  - `valuation`: mode, direction, size and an optional half-life ≥ 14 years (§6.3);
  - `regime`: label and share, which become inequality constraints through the regime classifier's rules.
- **`events`.** Each narrated event lists its generic templates and channels, `generic_footprint`, and
  `how_decomposed`. An event without a generic footprint is recorded but is not an input (§7.2).
- **Never allowed:** asset prices or returns, valuation numbers (except the qualitative `c_q.valuation`), E3 fields,
  `y_*` legs, and, in test mode, any calendar year or month count.
- **Every field is a measured series.** Each QTable field is a card row with a measured proxy and a first quarter.
  Text-derived concepts with no measured non-price proxy (policy credibility, sentiment beyond valuation) are not
  fields; a narrative that relies on them gets the note 「没有可测的对应变量，未作输入」.
- **Generation.** The JSON schema is generated from the card (`qtable_json_schema()`). The prompt's schema section is a
  compact field list, not the 31 KB schema. A test checks that the field enum equals the card's x_q set.

### 3.5 Masking and completion

- **Exact conditioning.** The stated components are conditioned on within the factor layer's joint innovation
  distribution (§5.3):
  - hard values are met exactly;
  - soft values and qualitative sizes are met within their sd;
  - ranges are met by truncated sampling: accept/reject while the acceptance rate is at least 10%, otherwise a Gibbs
    step on the constrained coordinates.
- **Saturation.** Before conditioning, the standardized gap between a stated input and its conditional mean saturates
  at ±4 training sd. This is fixed a priori, as in the qmodel. Mechanical own-price inputs (d_oil for crude, d_y10 for
  the treasury) pass through exactly. Each clipped field shows "超出历史范围，按 ±4 个标准差截断".
- **Completion shrinkage.** In quarters where no E1 / E1c / E2 field is stated, the completed mean of those fields is
  shrunk toward the state-conditional unconditional mean. The weight w ∈ [0, 1] is chosen by CRPS on validation
  origins up to 2014Q4, and the variance is mixed accordingly. The reason: completing market prices from macro alone
  has non-positive out-of-sample value and had the wrong sign in 2022 and 2007–09 (E14). The model must not mislead
  where it cannot infer.
- **Regime-dependent covariance** (ZLB, core > 4%), shrunk toward the pooled covariance: a candidate decided in T-C
  (§5.7).
- **Plausibility.** Every statement set gets a rarity statistic: the Mahalanobis norm of the innovation adjustment the
  conditioning needs, per quarter, as a χ² percentile. Above the 99th percentile the warning reads
  "这个组合在历史上极少见（需要约 X 个标准差的冲击）". For comparison, the joint study's stagflation set needed 0.81 sd
  per constraint. This is the agent tree's "rarity" capability.

### 3.6 Rollout over 10–15 years, and the dynamics engine

**Path mode** is used for scenarios, the lab, the narrative test and continuations. The factor layer draws the whole
factor path for H quarters at once, conditioned on every statement in the QTable, including those about later quarters.
Conditioning is therefore two-sided: a statement about 2029 shapes 2027–28.

The dynamics engine that fills unspecified quarters is the factor layer itself (§5):
- its BVAR dynamics, pulled toward the long-run anchors;
- a shadow policy rate under the zero bound;
- the cycle module, in quarters where the scenario says nothing about growth;
- bootstrapped whitened innovation vectors.

The asset layer then runs quarter by quarter and rolls the states:
- levels from changes (ff, y10, spreads, unemployment, real oil, the dollar);
- the inflation trend (EWMA, 36-month half-life) and the Sahm gap;
- `s_zlb` and `s_core4`;
- the valuation and income states (§6).

After the scenario horizon, the continuation (used by portfolio and withdrawal up to 40 years) runs the same model
unconditionally toward the long-run anchors.

**Statement grain.** The user's quarter-by-quarter reading holds exactly: each quarter is the cell of §3.2 with s_q from
the simulated past. Path mode adds only that future statements are known when the past is drawn. A one-step cell cannot
do that.

**Monthly paths.** The quarterly stack produces no monthly paths. Charts are quarterly. Portfolio and withdrawal take
quarterly sets (§11.3). A disaggregation utility splits each quarter with a kNN-matched historical quarter's monthly
profile and keeps quarterly totals exact. It exists only for monthly consumers during the transition, and its output is
labelled 季内月度路径为历史季度形态拆分（假设）.

### 3.7 Step mode

Step mode is one-sided: one quarter's conditional draw, given the state and the given values. The agent tree needs this
(it branches every quarter, so the future is unknown), and the lab's 逐季 view uses it.

Consistency test: path mode with statements only in quarter 1 must give quarter 1 the same distribution as step mode
(two-sample KS p > 0.01 on 10,000 draws per field).

### 3.8 Long-run assumptions (anchors)

- **Forward scenarios.** The anchors are the scenario's 长期假设. The defaults are the joint study's C′ anchors, which
  agree with today's engine: CPI 2.5, services CPI 3.0, real policy rate (r*) 0.75, real 10-year 1.5 (term premium
  0.75), Baa−10y spread 2.0, unemployment 4.8, real oil and the real dollar at their training medians. All are labelled
  判断值. The user edits them in one place, and the agent tree's U-9 points to the same place.
- **Backtests and evaluation.** The anchors use only data up to the train end, by the pre-registered rule of §5.5.

### 3.9 Package layout, files, build steps, endpoints

```
scenariolab/qmodel/
  config_q.py      quarterly-stack constants (kept out of config.py to avoid merge conflicts)
  card.py          model card: load, validate, qtable_json_schema(), prompt sections        card.json (generated)
  panel.py         quarterly panel (ports qmodel/qdata.py and valuation/valblock.py)
  policy.py        reaction functions, stabilizers, shocks u_* (ports qcore's policy split)
  factors.py       BVAR core + satellites, conditioning, ZLB, cycle module, completion shrinkage
                   (ports joint/jq.py, jq2–jq4, j11; qmodel/qcore.py x-block EM and FA covariance)
  assets.py        L0, T-A candidates (flagged), residual model, income legs
  valuation.py     identities, legs, valuation modes, implied CAPE and P/D paths
  rollout.py       path mode, step mode, state roll Φ, continuation, monthly disaggregation (export only)
  qtable.py        QTable parse/validate; to constraints; to the monthly engine's phase spec
  adapter.py       make_adapter() for the agent tree; CoreView support for instruments
  spec.py          frozen spec: version, data/code/card hashes, pre-registration records
  api.py           APIRouter mounted before the static route
scenariolab/evaluation/
  harness.py       port of simharness: conditional paths, metrics, placebo-corrected anatomy
  protocol.py      pre-registered protocols and gates T-A, T-F, T-C, T-S, T-N (one place)
  preregistered/   frozen JSON records (append-only; a test refuses edits to a committed record)
  narrative/       episodes.py, prompts.py (blind builder), runner.py, score.py, audit.py, report.py, sep.py
  ledger.py        prospective ledger (quarterly)
data/processed/panel_quarterly.csv, panel_quarterly_meta.json
data/results/qmodel.json (fit summary, acceptance results), evaluation.json, narrative/<run_id>/…, ledger/*.jsonl
data/evaluation/narratives/<code>/{v1,v2}/{n1,n2,n3}.md;  data/evaluation/key.json (sealed: read by the scorer only)
tools/build_model_card.py, tools/schema_view.py (the blind schema view), tools/qtable_from_yaml.py
```

**Build steps** (`python -m scenariolab.build --only …`):
- `quarterly`: the panel, about 10 s.
- `qmodel`: fit plus the acceptance statistics of T-F and T-C, about 2–5 min.
- `evaluation`: opt-in. It reproduces the frozen T-A and T-S records and recomputes the conditional-path tables, about
  10–20 min.
- `narrative`: mock backend and stand-in scoring only. It never calls a real LLM.
- `forecast`: gains the scenario mixture (§9).

**Endpoints** (Chinese error details, as today):
- `GET /api/q/card`
- `POST /api/q/run {qtable, n_paths, seed, mode}`, which starts a job
- `GET /api/q/run/{id}`
- `GET /api/evaluation`
- `GET /api/evaluation/narrative/runs`
- `POST /api/evaluation/narrative/estimate`
- `POST /api/evaluation/narrative/run {estimate_id, confirm: true}`
- `GET /api/forecast/scenario_mix`

---

## 4. The asset layer, and what is adopted

### 4.1 Default: L0

- **F.** F is the factor round's model in `data/results/model_selection.json` on `feat/causeway-v2`:
  - factors oil, ip, cpi_goods, cpi_services, ff, y10, spread, usd, unrate, gz and tp10_x_core4;
  - energy also takes metals_oil, and gold is fitted from 1982-01;
  - a monthly OLS on log price returns, refitted at every train end on monthly data up to the train end.
- **Quarterly prediction.** r̂_{a,q} = 3·α_a + Σ_k β_{a,k}·F_{k,q}, where F_{k,q} is the quarter's sum of the
  monthly factor changes, obtained from x_q by unit conversion:
  - d_oil/100 → oil; d_ip/100 → ip; d_cpi_goods/400 and d_cpi_serv/400 → the CPI factors;
  - d_ff, d_y10, d_spread, d_gz, d_unrate unchanged (pp); d_usd/100 → usd;
  - tp10_x_core4 = d_tp10 · 1{s_core4}. The start-of-quarter switch approximates F's monthly switch; it is documented,
    and its effect is measured in the harness reproduction test.
- **Evidence for L0:**
  - 1Q full x: +0.126 / +0.261 / +0.441 (split); pooled rolling +0.226 [+0.12, +0.32] (n 167);
  - the quarterly ridge shrunk to the monthly betas (L1-Fp) gives 0.137 / 0.255 / 0.439, the same thing;
  - the quarterly fit without the prior gives 0.141 / 0.154 / 0.358;
  - the pure quarterly OLS for us_equity gives −0.72 at 12Q.
- **Mechanical assets are identities, never regressions:**
  - crude oil = d_oil (WTI spot, no roll yield, as today);
  - the treasury = par-bond repricing on the y10 path, with coupon y10/4;
  - cash = the bill path.

  They are scored separately from the 10 risk assets, because their inputs are their own prices (critic on N2).

### 4.2 Candidates and the pre-registered test T-A

The candidates, frozen in `evaluation/preregistered/<date>_TA.json` before any run:

| Id | Candidate | Evidence so far (selected on which data) |
|---|---|---|
| C1 | tier-1 fields beyond F: d_igrea, x_supply_chain, d_bdlev, d_epu, d_tp10 (uninteracted), the oil-cause split d_oil_sup; scoped fields d_corptax, d_sent (financials only), d_usd_em, d_natgas, d_lprod, d_tips10 (2003+) | schema tiers, chosen with all three decades (admitted) |
| C2 | states and interactions: s_vol, s_zlb, s_core4; Δy10 × P/D; oil × ZLB (the qmodel's L2 set) | qmodel q03/q04, chosen on the test decades; Δy10 × P/D worth +0.063 mean10 and +0.374 for nasdaq in 2005–14 |
| C3 | next-quarter ip and unrate (S+Lc), in path mode only | layers, chosen from full-sample lead–lag evidence |
| C4 | V-AGG: us_equity as the cap-weighted sum of 8 sector equations | schema v04; matches the direct equation (report-only) |
| NN, NN-res | masked MLP; G plus an MLP correction | qmodel; report-only (§4.3) |

**Protocol.**
1. **Selection** uses rolling origins 1990Q1–2014Q4 with a yearly refit on data to the previous December, at 1Q and 4Q,
   with the ported harness metrics. A candidate passes the gate if its gain over L0 is at least +0.005 in both
   1995–2004 and 2005–14, is not ≤ −0.02 in either, and has a 90% lower bound above 0 in one of them. This is the
   schema's tier rule restricted to the two decades before the test.
2. **Test.** Train to 2014Q4 and score 2015Q1–2026Q3 once: 47 quarters at 1Q and 11 non-overlapping 4Q windows. The
   selected additions are adopted only if:
   - ΔR²(1Q) against L0 has a 90% block-bootstrap lower bound above 0 (blocks of 4 quarters, 2000 draws);
   - ΔR²(4Q) has a lower bound above −0.02;
   - 80% coverage at 1Q is within [0.75, 0.88].
3. **Disclosure, printed with the result.** 2015–26 has been seen by every earlier round: the factor round's test was
   used for its choice, the schema tiers used it, and the qmodel scored its variants on it. This is a pre-registered
   re-test, not a fresh holdout. The only clean holdout is the prospective ledger from 2026Q4 (§9).
4. **If T-A fails or has not been run, the asset layer is L0.** Fields outside F then reach assets only through the
   factor layer. For example, a stated d_epu shifts the completed d_gz, which carries a beta.

### 4.3 The neural network

**What was built.** A masked MLP with inputs [s, x ⊙ m, m] and outputs [r, x]: one tanh layer with a linear skip, an L2
penalty chosen by time-ordered validation, L-BFGS on mask-augmented data, and four seeds per fit.

**Results.**

| Split | NN | G | NN − G [90% CI] |
|---|---|---|---|
| 1995–2004 | −0.062 | +0.152 | −0.214, excludes 0 |
| 2005–14 | +0.004 | +0.226 | −0.222, excludes 0 |
| 2015–26 | +0.185 | +0.492 | −0.307, excludes 0 |

- Pooled rolling 1Q: NN +0.084 against G +0.222.
- The seed sd of its R² is 0.011–0.014, so the loss is not seed luck.
- NN-res stays within ±0.05 of G and is never significant; its validation picked heavy penalties in 2 of 3 splits.

**Decision: not adopted.** The NN did not beat the linear or Gaussian versions in any split. It stays a report-only row
in T-A under the same protocol. It may re-enter only if it beats both G and L0, with a lower bound above 0 at 1Q and 4Q,
in both validation decades and in the test.

The likely reason it loses: about 170 training quarters for 57 inputs and 13 outputs. The UI's method text says this in
one line.

### 4.4 Drift, the CMA and attribution

- **Drift.** α_a comes from the fit.
- **Forward scenarios.** For equities, the training-period re-rating of the CAPE multiple is removed:
  α_a,fwd = α_a − ρ_a·m̄.
  - m̄ is the mean quarterly Δlog CAPE multiple over the training sample: +1.18 / +1.80 / +1.47 / +1.89 % per year from
    1973 to the 1994 / 2004 / 2014 / 2026 ends [d05].
  - ρ_a = 1 for us_equity; sectors and nasdaq use their beta to the market, as the CMA scales its market adjustment.
  - Default on (U-5).
  - The sector relative-drift shrink and gold's real-drift removal stay. There is no reversion term: the engine's
    `valuation_state` CMA method with reversion 0, which the valuation report recommends.
  - The P/D re-rating is not used, because about 30% of it is payout decline (critic 8).
- **Backtests and the narrative test** report both CMA arms ("historical" and "no re-rating"), pre-registered.
- **Attribution.** For every asset, quarter and path, the return is split linearly into:
  - the fields the QTable stated;
  - the fields the model completed;
  - drift (α and the CMA);
  - the valuation statement's shift;
  - the residual (mean 0).

  Every scenario detail shows the medians as 情景说了什么 / 模型补了什么 / 历史常态漂移 / 估值假设. This is critic 6
  made visible: in E05 and E12 the drift carried +0.20 and +0.27 of the median.

### 4.5 Residuals and bands

- **Residual vectors.** The 13-asset quarterly residual vectors from training quarters are bootstrapped in 2-quarter
  blocks; blocks of 1 and 4 are pre-registered sensitivity arms.
  - They are uncoupled from the factor innovations, because OLS residuals are orthogonal to their regressors. Coupling
    would bring a drawn quarter's equity residual along with a factor shock that conditioning has changed.
  - Each asset's residual is scaled by exp(γ_a · (log s_vol − mean)), with γ fitted on training, as in the qmodel.
- **Parameter uncertainty.** 200 block-bootstrap refits of the monthly F (12-month blocks); each path uses one.
- **Calibration target.** 80% coverage of 1Q and 4Q returns within [0.75, 0.88], pooled rolling (167 and 41 windows).
  12–20Q coverage is reported with the anecdote label: with 2–3 windows per decade, 0.55 cannot be told from 0.80
  (critic, PARTLY on the harness band).
- **Draw dates.** Every path records the historical quarter label of each residual vector it drew. The instruments
  layer needs it for pairing (§11.2).

### 4.6 The switch gate T-S: when the quarterly stack becomes the default

The quarterly stack replaces the monthly engine as the default for the 13 scenarios, the forecast tab, portfolio and
withdrawal only if all of these hold, on the narrative test's clean episode set (§8.2):

1. **T-F passes** (§5.8).
2. **Oracle arms O-A, O-B and O-full:** Q − E equity9 MAE has a 90% upper bound below +0.02, both at 4Q and at the
   window end. E is the monthly engine running the selected model F, not the 'base' model the research round used.
3. **Calibration:** the mean |PIT − 0.5| of Q is at most E's + 0.02.
4. **Narrative arms N1–N3 on identical translations** (the QTable converted for E, §7.4): Q − E has an upper bound below
   +0.03.

Until then the monthly engine stays the default. The quarterly stack serves the lab and the agent tree, labelled 试运行.
The evaluation tab shows which condition failed, with its numbers.

The bar is non-inferiority with a tight margin, not superiority: the quarterly stack must be shown not to be worse than
the monthly engine by more than 0.02 (oracle arms) or 0.03 (narrative arms). The research qmodel would fail it. Its
oracle upper bound was +0.021, and on narratives it lost by +0.056 to +0.067 with every lower bound above 0 (E20). The
causes were poor completion of financial conditions and overconfidence (PIT 0.30–0.32 against 0.25). The redesign
addresses both (§3.5, §4.5, §7.1), and T-S checks that it worked.

---

## 5. The factor layer: one joint model

### 5.1 Variables

**Core** (BVAR, quarterly levels or rates; the joint study's B3 set):

| Symbol | Meaning |
|---|---|
| pc | headline CPI inflation, annualized |
| ps | services CPI inflation |
| ip | IP growth, annualized |
| ro | 100·log real oil |
| rff | real policy rate, as a shadow: ff − π_trend |
| ry10 | real 10-year: y10 − π_trend |
| sp | Baa − 10y, timing-fixed |
| gz | the GZ spread |
| usd | 100·log broad dollar |
| lu | log(u − 2): unemployment, transformed so u > 2 |

π_trend is an EWMA of headline CPI with a 36-month half-life. Goods inflation comes from the identity with
training-period weights (`cpi_weights`).

**Satellites.** Every other x field (Appendix A: "只经由因子层" and the candidates). Each has the form
w_q = d + ρ·w_{q−1} + Λ·k_q + Π·s_q + e^w_q, where k_q is the core vector above.
- Fitted by ridge with blocked 5-fold CV on training quarters.
- The innovations share one covariance with the core: factor-analytic with k ∈ {3, 5, 8} plus a diagonal, or shrinkage.
  The form is chosen by masked-entry error on the last training block, as in qcore.
- Missing-data EM: late series are masked before they start, never invented (d_y2 1976Q3, d_infexp1 1978Q2, d_hpi
  1975Q2, x_lending_std 1990Q2, d_infexp_lr 1990Q3, d_epu_health and d_chargeoff 1985Q2, d_tips10 2003Q2).

**Policy shocks.** All are fitted to the train end on current-vintage data (a caveat: ALFRED re-estimation is an open
question, §15):
- u_ff: the reaction function on same-quarter E0 macro and the states, non-ZLB quarters only (R² 0.54 / 0.51 / 0.46 at
  the three split ends); u_ff = 0 at the ZLB.
- u_bs: from 2008Q3, ≡ 0 before.
- u_def and u_tr: automatic stabilizers on the same-quarter and previous-quarter Δunemployment.
- u_ptax: includes the last four quarters' equity returns (capital-gains taxes).

**Mapping to schema units** (exact; tested):
- d_ff = Δ observed nominal ff;
- d_y10 = Δ(ry10 + π_trend);
- d_spread = Δsp; d_gz = Δgz; d_usd = Δusd; d_unrate = Δu;
- d_ip = ip/4;
- d_cpi_* are the rates themselves;
- d_oil = Δro + 100·Δlog CPI.

### 5.2 Estimation

- **Core.** BVAR(2) with Minnesota and sum-of-coefficients dummy priors (Bańbura, Giannone & Reichlin 2010), as in
  `jq.py`:
  - λ by marginal likelihood on the training window only (0.25–0.35 in the joint study);
  - a steady-state prior on the unconditional means (Villani 2009), centred on the anchors of §5.5;
  - 100 posterior draws; explosive draws are redrawn and the rejection rate is recorded (22% in the joint study).
- **Robustness without dates.** The joint study excluded 2020Q2–2021Q2. That exclusion is replaced by Student-t
  likelihood weights, w_t = (ν + K)/(ν + m²_t), estimated by EM, with ν ∈ {4, 8, 16, ∞} chosen by marginal likelihood
  on training. Extreme quarters (m² ≈ 1,127 for 2020Q2, 302 for 2008Q4, 441 for 1980Q4 against a robust covariance
  [d02]) get small weights by a rule, not by a calendar. No dated exclusion or weight remains in the quarterly stack.
- **Simulation tails.** Simulation draws whitened residual vectors with equal probability, i.e. at history's own
  frequency: a 2020Q2-size quarter appears in about 17% of 10-year paths (1 in 214 quarters). The monthly engine's dated
  `RARE_MONTHS` weight is not copied. U-12 lets the user lower tail frequencies explicitly in the scenario layer.
- **Satellites.** Point estimates in v1, with parameter uncertainty only from the core draws and the asset-layer
  bootstrap.

### 5.3 Conditioning, the do-operator, inequality statements

- **Statements become linear restrictions R z = r on the stacked path:** levels, changes, sums, means and y/y. For CPI
  y/y in the first quarters, known history fills the lags.
- **Hard statements** are imposed exactly by projecting the stacked innovations
  (e* = e + V G′ (G V G′ + Ω)⁻¹ (r − R μ − G e), with G = R M and V = I ⊗ Σ; Waggoner & Zha 1999). This is the joint
  study's implementation: met to 5e-14 on every path, and the Monte Carlo mean equals the analytic conditional mean.
- **Soft statements** add Ω from the stated sd (Andersson, Palmqvist & Waggoner 2010).
- **Bootstrapped innovations** keep their empirical shape in the unconstrained directions.
- **Ranges and regime requests** use truncated sampling (§3.5).
- **The do-operator.** Variables are ordered recursively within the quarter: [E0 macro] → [E0p policy] →
  [E1c, E1, E2]. A stated u_* is a structural shock in its policy equation:
  - same-quarter E0 macro is not conditioned on it;
  - E1 and E2 respond on impact through the lower-triangular block;
  - E0 responds with the VAR's lags.

  A stated raw policy field (d_ff without u_ff) is a reduced-form observation and imputes the macro associated with
  it. The translator rules of §7.3 decide which form the text calls for: 「因为衰退而降息」 is written as the recession's
  macro; 「紧急降息」 is written as u_ff. This is the agent tree's "structural" capability (its open question 1). Its
  fiscal response table can be replaced by u_def and u_tr.
- **QE.** u_bs alone moves no asset. The effect cannot be learned (26 quarters, 4 programs; raw d_bs gave equities the
  wrong sign). A QE statement must come with its intervention paths on d_tp10, d_y2 and d_gz (template G08). The
  validator warns when u_bs is stated without them.

### 5.4 The zero bound and bounded variables

- **Shadow rate.** The VAR runs on the shadow rff. Observed ff = max(rff + π_trend, 0.1).
  - Estimation: the ff equation uses non-ZLB quarters only, with ZLB quarters missing in the EM. A pre-registered
    alternative is Johannsen & Mertens (2021) data augmentation with shadow ≤ floor.
  - This removes the plain BVAR's 2009 misses of about −3.5 pp; the 1Q fill improves from −2.9 to −1.0 (joint).
- **Unemployment.** lu = log(u − 2). Acceptance requires u ≥ 2.5 always and u < 3% in at most 5% of simulated quarters.
  Every generator in the joint study put u < 3% in 12–20% of quarters; history's quarterly minimum since 1973 is 3.5
  [d01].
- **Spreads.** Floored at 0.2 by a log transform of (spread − 0.1) if the floor binds in more than 0.5% of draws.

### 5.5 Anchors fitted to the train end

**Backtests and evaluation** (pre-registered rule, `engine/anchors.py`, shared with the monthly engine's backtest in
P0.1):
- r* = trailing 40-quarter mean of (ff − core CPI y/y);
- term anchor = trailing 40-quarter mean of (y10 − ff);
- spread anchor = trailing 40-quarter mean of Baa − 10y;
- unemployment anchor = trailing 80-quarter mean;
- π_lr = the Michigan 5–10-year median at the train end from 1990Q2 on, before that the trailing 40-quarter mean of core
  CPI y/y.

| Train end | r* | Term | Spread | Unemployment (80Q) | Today's fixed values used in every window |
|---|---|---|---|---|---|
| 1994Q4 | 2.19 | 1.67 | 2.01 | 7.05 | 0.75 / 0.75 / 2.0 / 4.8 |
| 2004Q4 | 1.79 | 1.24 | 2.26 | 5.73 | same |
| 2014Q4 | −0.35 | 1.73 | 2.71 | 6.01 | same |
| 2026Q2 | −0.74 | 0.46 | 2.04 | 5.77 | — |

The numbers are from [d02, d05]. The rules are crude, which is why they are pre-registered rather than tuned. A
pre-registered alternative for unemployment is CBO NROU real-time vintages from ALFRED, where they exist.

**Forward:** §3.8. The steady-state prior has sd 0.5 pp forward (the assumption binds) and 1.0 pp in backtests (the data
can move it).

### 5.6 Cycle module (the C′ hybrid)

**When it applies.** In every quarter where the QTable states nothing about growth (no d_ip, d_rgdp, d_unrate, regime
request or growth constraint).

**What it does.** The IP growth path is drawn from 2-quarter blocks of training-period quarterly IP growth, with equal
prior weights. When the scenario states regime odds, the weights are entropy-tilted toward them. The path is imposed as
a soft constraint (sd = 0.5 × the quarterly IP sd) on the BVAR's ip, so the other variables co-move with it.

**Why.** A linear VAR cannot produce short, sharp contractions. The joint study measured the share of contractions longer
than 6 quarters: plain BVAR 55%, C′ 22% (mean length 4.6 quarters, 14.1% of quarters in contraction). History has 0 of
7 such contractions (mean 4.1, 13.6%).

**Alternative.** Eight-quarter blocks of whitened innovations without the module, pre-registered and compared in T-F. The
better posterior predictive fit wins.

### 5.7 Regime-dependent covariance and completion shrinkage

- **Regime covariance.** Two regime indicators (`s_zlb`, `s_core4`). Σ_r = (n_r Σ̂_r + n0 Σ̂_pooled)/(n_r + n0) with
  n0 = 40 quarters. It addresses the qmodel's open question 1: at the zero bound, the pooled covariance infers rate cuts
  that cannot happen.
- **Completion shrinkage** (§3.5). The weight w is chosen on validation and frozen.

Both are decided in T-C.

### 5.8 Acceptance tests T-F and T-C

**T-F: posterior predictive checks** (`protocol.py`).
- Fits at 1994Q4, 2004Q4, 2014Q4 and 2026Q3, each run 52 quarters forward unconditionally, 2,000 paths.
- Every statistic is computed per simulated 13-year path. History's value must lie within the simulated [p05, p95],
  using non-overlapping history windows; the critic showed that overlapping ones inflate the sample.
- Hard checks are exact.

| Id | Statistic | History value (source) | Pass rule |
|---|---|---|---|
| F1 | delivery of stated values | — | hard: \|error\| < 1e-8 on every path; soft: 68% ± 5% inside ±1 stated sd |
| F2 | share of quarters with CPI y/y > 4%; 4-quarter autocorrelation of CPI y/y; mean high-inflation spell | 32.6% (n 215); non-overlapping 13-year windows 0.56 / 0.49 / −0.24 / 0.46; 7.8 quarters | within [p05, p95] |
| F3 | contractions per 15 years; mean length; share longer than 6Q | 1.95; 4.1Q; 0 of 7 | first two within [p05, p95]; share > 6Q ≤ 25% (P(0 of 7) ≥ 0.13) |
| F4 | bounce-back: IP growth over 4Q on the drawdown from the 8Q peak | −0.48 [−0.82, −0.30], 1960Q2–2026Q3 sample | within [p05, p95] |
| F5 | co-movement: Δff on Δu; Okun; corr(d_gz, us_equity); corr(d_spread, us_equity); stock–bond correlation by s_core4 | −1.44 [−2.02, −0.88]; −0.041 [−0.047, −0.034]; −0.58 and −0.60 (1985–2026, n 167) [d01]; regime-dependent | within [p05, p95]; the two credit correlations are the instruments' T18 |
| F6 | bounds | ff ≥ 0; u ≥ 3.5 since 1973 | ff ≥ 0.1 always; u ≥ 2.5 always and < 3% in ≤ 5% of quarters; CPI y/y > 15% in ≤ 0.5% |
| F7 | fill-in skill with CPI, IP and oil stated (rolling 1990–2026, 4Q/12Q): R² of unstated factors against no change | the engine's numbers in the joint study | ≥ the engine on unemployment and the Baa spread; ≥ −0.05 elsewhere; d_usd held at no change when unstated (no skill anywhere) |
| F8 | 80% coverage of 4Q factor changes (pooled, 41 windows) | — | within [0.72, 0.88] |
| F9 | do-operator | literature signs: a surprise easing raises equities, lowers d_y2, weakens the dollar; unemployment responds with a lag (peak at 4–8Q) | a stated −1 sd u_ff leaves same-quarter E0 exactly unchanged; signs as stated; a stated raw cut in a narrative that also states no recession imputes nothing |
| F10 | step and path consistency | — | §3.7 |

**T-C: completion tests.**
- Selection of w and the regime covariance on rolling origins up to 2014Q4; test on 2015Q1–2026Q3.
- Masking patterns: (a) a random 50% of x; (b) narrative-like, only E0 and E0p stated; (c) only financial conditions
  stated.
- Metrics per field: completion R² against unconditional, CRPS and 80% coverage.
- Adoption: the regime covariance is adopted if it improves E1 completion CRPS in ZLB quarters on validation without a
  loss elsewhere greater than 0.01.
- Gate: under pattern (b), every E1 field's completion R² on the test is ≥ −0.02. Where the model cannot infer, it must
  stay near unconditional.

---

## 6. The valuation block

### 6.1 The identity and the measures

**Multiple.** log P_q(CRSP) = log M_q + log CPI_q + log E10r_q. M is the CAPE multiple on the CRSP price
(`us_equity_lm_cape` in `valuation_quarterly.csv`; its level is arbitrary). E10r is the 10-year average of S&P real EPS
(Shiller 2026 workbook), and CPI is Shiller's.
- y_us_equity(price) = y_mult + y_fund, with y_mult = Δlog M and y_fund = Δlog CPI + Δlog E10r.
- y_fund is smooth: mostly CPI pass-through (coefficient 0.97, t 8.9, R² 0.56; valuation report).

**Why the CAPE multiple and not P/Ê** (log changes [d01]):

| Window | price | CAPE multiple | P/D | P/Ê (schema) |
|---|---|---|---|---|
| all quarters (corr with price; share of variance; n 213) | — | 0.990; 1.011 | 0.964; 0.973 | 0.647; 0.979 |
| 2007Q3–2009Q2 | −0.468 | −0.492 | −0.460 | **+1.005** |
| 2001Q1–2002Q4 | −0.383 | −0.469 | −0.357 | **+0.256** |
| 1995Q3–2000Q1 | +1.008 | +0.668 | +0.789 | +0.621 |
| 2022Q1–Q3 | −0.296 | −0.408 | −0.346 | −0.457 |

The schema's P/Ê leg rises in both busts because of write-downs. The CAPE leg does not.

**Consequence.** The multiple carries almost all the price variance. A stated multiple path is therefore a stated price
path, and the design treats it as one (§6.3). This is critic 2 applied: the decomposition is bookkeeping, not evidence.

**Income leg.** CRSP P/D is exact and carries dividends: income = D/P per quarter. Dividend growth uses the training
equation (CPI 0.44, own lag 0.30, R² 0.13), and the P/D path is an output, y_pd. The P/D is not used for re-rating,
because about 30% of its 1973–2025 rise is payout decline (critic: 0.26 of 0.88 log).

**Display only, with caveats:** trailing P/E (`s_pe`) and the NFC FCF yield proxy.

**The monthly engine's ValuationModel** (`fit_valuation`, off by default) stays off. Its pull toward a fair value is one
of the pulls that lost out of sample (§6.2).

### 6.2 States and today's values

| State | Source and timing | Today (known at the start of 2026Q4) | Use |
|---|---|---|---|
| CAPE | Shiller 2026 workbook; E lagged 3 months | 41.2 (97.8th percentile since 1948) | initial condition; reference for valuation statements; implied path; the T-A interaction candidate Δy10 × P/D |
| ECY = 100/CAPE − real 10y | Shiller | 0.79% | display; the only measure with positive 10-year out-of-sample R² after 1950 (+0.29 on 7 decade blocks, an unreliable interval) |
| CRSP P/D | Ken French; lag 0 | 97.4 | income; y_pd |
| NFC FCF yield proxy | NIPA/Z.1, release-dated | 1.96% (44th percentile) | display only, labelled 全部非金融公司的代理，不是 SPY/QQQ 的自由现金流收益率; it disagrees with the NFC profit yield (2.71%, 3rd percentile) by 41 percentile points |
| BusEq P/B | French annual book equity, used from July | 12.2 (2000Q1: 15.1) | display; the only continuous tech valuation |
| Sector P/D, REIT P/D | French; Nareit | — | display |

**Not used as a predictor or a pull.** Every out-of-sample test lost:
- pooled 12/20Q R²: constant anchor −0.47 / −0.75; real-yield anchor −0.25 / −0.27; 20-year rolling anchor
  −0.32 / −0.56; a fixed 5% a year −0.10 / −0.12;
- seven starting-level states added to F all lowered 12/20Q R²;
- κ ≈ 0 after 1973.

### 6.3 Valuation statements in a scenario (`c_q.valuation`)

| Mode | Meaning | Implementation |
|---|---|---|
| `inferred` (default) | the multiple does whatever the macro and credit path implies | no constraint. y_mult = asset-layer price return − y_fund; the residual multiple has zero drift |
| `hold` | the multiple stays where it is over the stated quarters | Σ y_mult = 0, soft, sd = 0.25 × the 温和 bin |
| `rerate {direction, size}` | re-rating of a stated size | Σ y_mult = ±size, soft, sd = 0.25 × size. Sizes 温和 / 明显 / 极端 = p50 / p75 / p95 of \|Σ Δlog M\| over spans of the same length in the training sample. On 1973–2014 [d05]: 4Q 0.12 / 0.20 / 0.34; 8Q 0.15 / 0.29 / 0.49; 12Q 0.21 / 0.30 / 0.53; 20Q 0.30 / 0.44 / 0.62 |
| `reversion {half_life_years ≥ 14}` | explicit slow mean reversion | drift −κ (log M − log M*), with κ = ln 2/(4·HL) per quarter and M* = the training median; never a default |

**Implementation.** Per path, the us_equity residual is shifted so that the constraint holds. The other equities move
by their residual regression on the us_equity residual (the qmodel's method). Nasdaq and the sectors therefore inherit
the market's re-rating.

**UI wording:** 「估值假设：这一条相当于直接假设美股价格在这 N 个季度里 ±X%（其中基本面腿 +Y%）」.

**Narrative test.** Never stated in N arms; the validator refuses valuation statements in test mode. The realized
multiple appears only in the identity-bound arm V, labelled 恒等式上界（不是模型能力）.

**Translator.** 「估值泡沫」 and 「估值压缩」 map to rerate with a size. A P/E number is refused unless the user writes it
in the lab; it then becomes `src: stated` and the wording above applies.

### 6.4 Per asset (and the QQQ limitation)

| Asset | Valuation state | Multiple leg | Limitation |
|---|---|---|---|
| us_equity (SPY proxy) | CAPE multiple, P/D, ECY | y_mult and y_fund identities | CAPE is the S&P composite's; CRSP is the broader market (the identity uses the CRSP price over S&P earnings) |
| nasdaq (QQQ) | BusEq P/B (display) | the market's leg through the residual regression | **No free NASDAQ-100 P/E, earnings or FCF exists in the allowed sources** (Ken French, Shiller, FRED/ALFRED, NY Fed, federalreserve.gov, BLS, BEA). BusEq P/D breaks in 2003–05 (843 in 1999, 58 in 2004, 163 in 2007: Microsoft's special dividend). BusEq P/B is the only continuous tech valuation (BusEq price tracks NASDAQ with quarterly corr 0.95), and a block built on it lost to F. A tech-specific re-rating therefore has no measured series. The UI says 「纳指的估值只能用全市场 CAPE 近似」 |
| six sectors | own P/D, annual P/B (display) | market leg | lumpy dividends; the annual June/July reclassification moves P/D; per-sector blocks lost to F |
| REITs | Nareit P/D | market leg | the in-sample reversion (0.18/q) fails out of sample, so there is no pull |
| gold | real gold level (view) | none | no cash flow |
| crude | real oil level (factor-layer anchor) | none | WTI spot, no roll yield |
| treasury, cash | the yield itself | none | identities |

### 6.5 Data fixes

1. **s_cape.** Build it from the Shiller 2026 workbook (E and D to 2026-06) and retire MAIN's NIPA splice, which reads
   1.040–1.125 × Shiller since 2023-07 (43.72 against 40.58 in 2026-09). Today it feeds the production translator's
   state table. This is P0.3.
2. **The 2026H1 E jump.** Trailing E rises from 240.6 (2025-12) to 295.4 (2026-06). Whether these are reported or
   estimated values cannot be told from the file (open question §15). Until confirmed, the UI marks today's P/E and E10
   as 「E 的最近两个季度可能是估计值」.
3. P/Ê is display only. The P/D re-rating is not used. The FCF proxy is labelled. x_oil_cause is an enum.

### 6.6 How valuation results are reported

Never as corr(miss, Δmultiple)². That statistic is an identity: a constant forecast scores 0.854 across episodes and
0.83–1.00 on the harness. Report instead:
- (a) corr(predicted Δmultiple, realized Δmultiple) across windows or episodes, with n;
- (b) the share of the realized multiple change the model predicted: the slope of realized on predicted, and R² against
  zero;
- (c) a placebo row (constant forecast) in every table.

The harness port (`evaluation/harness.py`) computes all three, and the old anatomy columns are kept only in the export.

---

## 7. The scenario layer and the translator

### 7.1 What a scenario must state

Phases are whole quarters, and a phase may be one quarter. Per phase:

| Requirement | Content | If missing |
|---|---|---|
| R1 policy | a d_ff path, a u_ff intervention, or 「由反应函数决定」 (explicitly) | completed by the reaction function, marked |
| R2 growth | d_ip / d_rgdp / d_unrate, or a regime request | the cycle module (§5.6), marked |
| R3 inflation | d_cpi_goods / d_cpi_serv or a y/y constraint | completed, marked |
| R4 oil | d_oil plus x_oil_cause | completed, marked |
| **R5 financial conditions (now required)** | `fc.stance` at least; d_gz / d_spread numbers are optional | warning W-FC: 「金融条件未说明：信用利差由模型补全。历史检验里，补全的信用利差与无条件抽样相当，而资产结果主要取决于它」 |
| R6 valuation mode | inferred (default) / hold / rerate | inferred |

Optional: anchors, announced paths, satellites, constraints, regime.

**Why R5 is required.** It is the only input that carries out-of-sample skill (E1). Stating it is close to stating the
equity risk premium; that is the scenario author's hypothesis, and the UI says so. The evaluation reports
"macro only" (O-A, N1) and "macro + stated financial conditions" (O-B, N2) as separate arms (§8.3). U-2 lets the user
make R5 optional again.

### 7.2 The production translator (QTable) and decomposition into generic channels

**Prompt sections, in order:**
1. The role.
2. The model card's x_q and c_q fields, compact: id, 中文名, unit, timing, endogeneity, permission.
3. The size-bin table.
4. The generic templates G01–G35:
   - production mode: signs and sd sizes from all history;
   - test mode: **signs only**, never sizes, never the dated `measured` footprints or `footprint_quarters`.
5. The rules: decompose every event into generic channels; R1–R6; what is refused; quotes; quarter indices instead of
   dates.
6. The state table:
   - production: today's levels;
   - test mode: percentile bands such as 「高（历史 80–95 分位）」, because combinations of levels date an episode.
7. The output schema section.

**User message:** the narrative only. The answer then goes through repair, validation (§7.3) and the report.

**Decomposition (the user's 5th message).**
- Every event is decomposed into generic channels that have at least two historical instances (the templates).
- `events[].how_decomposed` says how. Example: 「疫情封锁 → 服务需求塌陷、劳动供给下降、转移支付、降息到下限、扩表、信用利差先升后降」.
- An event without a generic footprint becomes a note, not an input. Example: accounting fraud becomes a note on the
  earnings leg.

**Few-shot examples** are synthetic, built from templates (a demand recession, an oil supply shock, a banking panic),
with no years, names or dated events. This applies the critic's finding on 「Y0−14 年海湾战争」.

**Backend and caching:** the existing translator machinery (`MSL_LLM_BACKEND`, the `data/llm_cache/` key, concurrency
2), plus the mock backend (§8.5).

### 7.3 Validation rules and refusals (Chinese messages)

| Rule | Mode | Message (abridged) |
|---|---|---|
| unknown field; wrong form; quote not found in the narrative | all | 字段不存在 / 引文在叙事里找不到：这个值没有文字依据 |
| E1/E1c/E2 entry whose quote does not name that variable (each card row lists Chinese keywords, e.g. 信用利差 / 公司债利差 / GZ for d_gz); a size outside the five bins | all | 文字没有说的市场价格和调查值不能由翻译器决定 |
| E3 fields, asset prices or returns, y_* legs | all | 资产价格与估值倍数不是输入 |
| valuation number | test mode: refused; lab: allowed only as the user's own number | 估值假设相当于直接假设股价 |
| value beyond 1.5 × the historical range | production: clipped with a warning; test mode: refused | 超出历史范围 |
| u_bs stated without intervention paths on d_tp10 / d_y2 / d_gz | all, warning | 扩表对资产的作用学不出来，请写明它对期限溢价、2 年期收益率和信用利差的作用 |
| more than two of d_rgdp / d_lprod / labour input, or deficit components plus the deficit | all | the third is inferred; contradictions reported (schema rule 2) |
| R1–R5 missing | all, warning | 由模型补全（附证据句） |
| any 19xx/20xx token or month count in any output string | test mode | refused (blind protocol) |
| per-phase r_star override | not in QTable | anchors live in c_q with quotes, so a stated policy path no longer forces the 10-year target |

The monthly translator (phase/view specs) keeps working until the switch. In P0.4 it gains the endogenous-view warning
(spread, GZ, EBP, sentiment, claims, EPU; layers) and refuses VALUATION_VARS in test mode.

### 7.4 Built-in scenarios as QTables; conversion for the monthly engine

**From YAML to QTable.** `tools/qtable_from_yaml.py` converts the 13 YAML specs deterministically:
- phase views become soft statements on the matching quarterly fields;
- view confidence becomes the sd;
- spread views become an fc stance;
- regime keys become regime requests.

A human then reviews each QTable and adds the financial-conditions stance where the YAML has none.

Test: the converted QTable run through the quarterly stack delivers the YAML's view means within the monthly engine's own
delivery tolerance, and the delivery is reported per view. The monthly engine keeps the YAML until the switch.

**From QTable to the monthly engine's phase spec** (`qtable.to_phase_spec`). This drives the narrative test's engine
arms with identical translations, which removes the research round's confound of separate spec and table translations.
- Consecutive quarters with the same stated-field set merge into a phase of at least 2 quarters.
- Each stated field becomes a monthly view:
  - f_ff mean = d_ff/3 pp a month;
  - f_cpi_goods mean = d_cpi_goods/1200;
  - f_ip = d_ip/300; f_oil = d_oil/300; f_usd = d_usd/300;
  - f_unrate = d_unrate/3; f_y10, f_spread and f_gz = /3.
- An fc stance becomes f_spread and f_gz views at the size bins.
- A field without a monthly column is dropped with the note 「月度引擎没有这个字段」.
- Valuation statements go to the monthly engine only as VALUATION_VARS views, with the warning.

### 7.5 Probabilities (Layer 1)

- **Source.** The hand-set `prior_prob` of each scenario is the default, and the user can override it in the UI.
  The KEY_STATES posterior tilt (`scenarios/probabilities.posterior`) is off by default. It turns on only after the
  leading-macro states pass a pre-registered macro_skill test: beat both the unconditional and the naive forecast in
  both halves. The current evidence is PLAUSIBLE only: term spread → Δunemployment out-of-sample +0.13, real ff → Δff
  +0.17, n 36.
- **What can be checked:**
  - coherence: probabilities sum to 1, and the support;
  - the mixture's implied base rates (P(contraction within 4 and within 8 quarters), P(CPI y/y > 4% for ≥ 4 quarters),
    P(ff rises ≥ 1 pp)) against history and against the SEP (§8.3);
  - prospective scoring in the ledger (§9).
- **What cannot:** the calibration of 13 overlapping 15-year scenarios, which have one realization.

---

## 8. The end-to-end narrative test (permanent)

### 8.1 What it can and cannot show

**It can measure:**
- translation fidelity (stated against realized factors);
- the chain's conditional skill given a description;
- the error decomposition into hypothesis and simulation error;
- discrimination between the right and a wrong narrative;
- an engine comparison on identical translations.

**It cannot measure:**
- ex-ante predictive skill. The LLM knows history; recognition is measured, not prevented.
- busts against ordinary episodes with statistical power: there are 5 busts in the clean set.

With 14 episodes, the minimum detectable equity9 difference is about 0.02–0.03 log points. The UI prints it beside every
claim.

### 8.2 Episodes

The clean set has at least 56 training quarters (14 years), and it carries the headline. Values are log; `useq` is the
window's realized us_equity price return, DD its maximum drawdown inside the window [d02]. Busts are pre-registered as
DD ≤ −0.20.

| Code | Window | Train Q | useq | DD | Bust | Narratives |
|---|---|---|---|---|---|---|
| E11 | 1990Q3–1991Q4 | 70 | +0.17 | −0.18 | | v1 + v2 |
| N13 | 1987Q3–1988Q4 | 58 | −0.09 | −0.27 | yes | new (v2) |
| E07 | 1994Q1–1995Q2 | 84 | +0.14 | −0.06 | | v1 + v2 |
| E02 | 1995Q3–1997Q4 | 90 | +0.55 | 0.00 | | v1 + v2 |
| N14 | 1998Q1–2000Q4 | 100 | +0.28 | −0.17 | | new (nasdaq +0.86; d_gz Σ +2.47) |
| E12 | 2001Q1–2002Q4 | 112 | −0.38 | −0.46 | yes | v1 + v2 |
| E01 | 2004Q3–2006Q4 | 126 | +0.23 | −0.03 | | v1 + v2 |
| E05 | 2007Q3–2009Q2 | 138 | −0.47 | −0.63 | yes | v1 + v2 |
| N15 | 2010Q2–2012Q4 | 149 | +0.20 | −0.18 | | new |
| E04 | 2014Q3–2015Q4 | 166 | +0.03 | −0.08 | | v1 + v2 |
| E09 | 2017Q1–2019Q4 | 176 | +0.36 | −0.16 | | v1 + v2 |
| E08 | 2020Q1–2021Q2 | 188 | +0.33 | −0.23 | yes | v1 + v2 |
| E10 | 2022Q1–2023Q4 | 196 | −0.02 | −0.30 | yes | v1 + v2 |
| N18 | 2024Q1–2026Q2 | 204 | +0.45 | −0.06 | | new (the most recent; certainly recognised) |

The clean set has 119 quarters.

**Short-training set** (reported apart, never pooled into the headline):
- E06 1973Q4–1975Q2 (3 training quarters; leave-window-out only, look-ahead in fitting);
- E03 1979Q4–1982Q2 (27);
- N16 1977Q1–1979Q3 (16);
- N17 1983Q1–1984Q4 (40).

No windows overlap [d02]. E07/E02 and N14/E12 are adjacent.

**Episode registry.** Codes are shuffled. The code-to-window key is sealed in `data/evaluation/key.json`, which only the
scorer reads. A CI test asserts that the prompt builder and runner cannot import it.

**Writer protocol v2** (a writer round, for all 18):
- no years, names or places;
- events at the channel level (「一家大型金融机构倒闭」), not unique details such as GSE takeovers or two automakers'
  managed bankruptcies (critic on the writer);
- no asset prices, valuations, earnings, sector outcomes or firm outcomes;
- N1: macro and policy with rounded magnitudes and month timing;
- N2 = N1 + a 金融条件 section;
- N3 = N1 without digits but stating 「覆盖期约 N 个季度」, so N1 and N3 differ in magnitude only (the writer's open
  question);
- automated scans: years, digits in N3, and a banned-word list that includes sector and firm-outcome words.

The v1 narratives (research round) are kept unchanged.

### 8.3 Arms

| Arm | Given | Translation | Meaning |
|---|---|---|---|
| U | nothing | — | no-narrative baseline |
| O-A | realized E0, E0p (raw) and x_oil_cause | none | pure macro oracle |
| O-A′ | O-A + realized E1c (commodity prices) | none | "石油打仗" taken as given |
| O-B | O-A′ + realized E1 (financial markets: rates, term premium, spreads, dollar, funding and leverage, house and property prices) | none | macro + financial-conditions oracle |
| O-full | O-B + realized E2 (surveys and news indices) = every x field | none | upper bound of the factor-to-asset chain |
| V | O-full + realized y_mult | none | **identity upper bound, not a model result** |
| N1, N2, N3 | translated QTables (v2 narratives) | LLM | macro / + financial conditions / no numbers |
| N-wrong | the paired episode's N1, translated against this episode's prompt and run from this origin | LLM | discrimination: the right narrative must beat a plausible wrong one |
| N-synth | 8 synthetic episodes: narratives written from the model's own simulated paths at 4 origins; truth = the simulated path | LLM | translation fidelity with no recognition possible |
| H-SEP | the Fed's SEP medians (unemployment, GDP growth, PCE and core PCE inflation from late 2007; the fed-funds median only from 2012, before which the policy rate is left to the reaction function) at each SEP date, real-time from FRED/ALFRED, converted to a QTable **by code** (PCE → CPI by the training-period gap; GDP → d_rgdp; linear quarterly paths to the year-end medians). The exact series ids and vintages are confirmed in P7 (`sep.py`), and the download needs the user's approval | none | a truly ex-ante hypothesis: no hindsight, no LLM; about 18–19 independent annual windows |
| H-SEP+FC0 | H-SEP with fc stance 中性 | none | the same hypothesis plus "no credit shock" |

- **N-wrong pairing (pre-registered, by code).** Each clean episode is paired with the clean episode of the nearest
  window length whose T0 is at least 5 years away, ties broken by code order. The pairing is frozen in the registry.
- **Engines.** Every arm runs on the quarterly stack (Q, with L0, and with G if T-A adopts it) and on the monthly engine
  (E, running F, through `to_phase_spec`). There are two CMA arms (§4.4).
- **Naming.** O-B here excludes the surveys and news indices (E2), unlike `schema.json`'s headline_test, so that
  financial markets are isolated; O-full adds E2.
- **Scoring rules (schema headline_test).** The treasury, crude and cash are scored separately. O-B − O-A′ is reported
  as "what the financial-conditions hypothesis carries".

### 8.4 Blind protocol and leakage controls

| Channel (critic 3, translator, writer) | Control | Automated check |
|---|---|---|
| T0 is derivable from the sample line plus the model start | the blind prompt has no sample line, no month counts and no model start | regex scan; the prompt hash is recorded |
| dated few-shot examples | synthetic few-shots only | scan for era names and event words |
| base-year unit labels (「Y0+22 年美元」) | none; real oil shown as a percentile band | scan |
| state levels date the episode | percentile bands in test mode | — |
| schema.json footprints (dated realized values in templates and `scenario_rules`; 354 year tokens) | the translator gets only the generated blind schema view (`tools/schema_view.py`): definitions, units, timing, permissions, template signs | the view's hash; scan for quarter and year tokens |
| shared context (the user's 2022–23 example; the lead's named episodes) | the translator process receives only the prompt and the narrative. The claude_code backend already runs with `--tools ""` in an empty temporary directory, so no repository, file or tool reaches it | the run manifest records every byte sent (prompt and narrative hashes) |
| unique events in narratives | writer protocol v2 | banned-word scan; recognition probe |
| recall of E1 values in N3 | in N3, E1 entries may only be size bins with quotes | contamination index; recall flag (corr > 0.9 with realized) |
| engine look-ahead (anchors, leave-window-out fits, e_* SVAR shocks, EBP, interpolated series, current-vintage r*) | train-end anchors; no leave-window-out in the headline; no e_* or EBP inputs; release-dated quarterly series; r* by the train-end rule | `check_context`: the episode's context is rebuilt from rows ≤ T0 only (truncation test) |
| translations of the same episode seeing each other | separate calls with no memory; N3 first; order randomised | manifest |

**Contamination index** per run:
- median |z| of the translated values against realized;
- baselines: the history prior's mean (0.66–0.76 in the research round) and the model's unconditional completion (0.42);
- the recognition rate, from a separate probe call per episode that asks 「这段叙事描述的是哪段时期？不知道就写 unknown」.
  It is scored against the sealed key and never fed back.

When recognition is 100%, the result is labelled 识别出年代的回放（不是盲测）. The test then still measures translation
fidelity and the chain, but not ex-ante skill.

### 8.5 Running it: mock, stand-in, real LLM

**Backends.**
- `mock` is the default under pytest, CI and every build step. It is deterministic:
  - for the 12 v1 episodes it replays the research round's stand-in QTables, with the x_oil_cause convention fixed and
    the fix recorded. These tables carry `src` codes but no quotes, so `import-standin` converts them to QTable/1 with
    `legacy: true`: the quote check is skipped, and every result from them is labelled 研究轮代译（无引文）;
  - otherwise it returns a minimal valid QTable: every field absent, fc 未说明.
  - Recorded fixtures cover malformed JSON, refusals, timeouts and usage-limit texts.
- `claude_code` is the real backend, as the user decided (USER_DECISIONS: local Claude Code CLI on the user's plan
  login; tests never spend tokens). It uses the existing `llm/claude_code.py`.
- `api` is used only when `MSL_LLM_BACKEND=api` and `ANTHROPIC_API_KEY` are set. It is billed per token, and the
  estimate dialog says so.

**Commands.**
```
python -m scenariolab.evaluation.narrative estimate [--backend claude_code|api] [--episodes clean|all|E05,N13]
        [--narratives v2] [--arms N1,N2,N3,wrong,synth] [--probe]          # prints calls, tokens, minutes; writes an estimate id
python -m scenariolab.evaluation.narrative run --estimate <id>             # asks 「确认运行？」 on a TTY; otherwise needs
        --i-understand-this-uses-my-plan (claude_code) or --i-understand-this-is-billed (api)
python -m scenariolab.evaluation.narrative score <run_id> | report <run_id> | import-standin
```

**Estimate for the default run** (all 18 episodes, v2, N1–N3, wrong, 8 synthetic, probes):
- 54 + 18 + 8 = 80 translations and 36 probes, 116 calls in all;
- about 20–25k input tokens and 3–8k output tokens per translation;
- 20–60 s per call at concurrency 2: about 20–60 minutes.

The default numbers are assumptions until the first run replaces them with measured ones.

**Run behaviour.**
- Cache: `data/llm_cache/` by (backend, model, effort, prompt hash, message hash), as today. Re-running is free;
  `--refresh` starts a new run id.
- Usage limits: the run pauses on the claude_code backend's usage-limit error and resumes by run id.
- UI: the evaluation tab's button 「用真实大模型重跑叙事检验…」 opens a dialog with the estimate
  (`/api/evaluation/narrative/estimate`). Confirming starts a job on the same queue as `/api/backtest/run`; progress
  streams; results appear when scoring finishes.
- **Never automatic.** No build step, test or CI job calls a real backend; a test asserts the mock is selected under
  pytest. The server refuses `/run` without `confirm: true` and an estimate id issued in the last 10 minutes.

**The research round's run** is imported as 研究轮（代译，不盲，2026-11-01). Its statistics are recomputed with the
critic's corrections, its numbers are kept, and its 36 translations are not re-run (USER_DECISIONS).

### 8.6 Scoring, statistics, and the claims the UI may make

**Measures:**
- cumulative log price return (cash: total) per asset and for the groups equity9, gold, crude, treasury + cash;
- horizons 1Q, 2Q, 4Q, 8Q and the window end;
- median error, MAE, CRPS, PIT and 80% coverage;
- corr(median, realized) across episodes;
- factor delivery: stated fields against the simulated median; unstated fields' completion against realized.

**Statistics:**
- The episode is the independent unit. Bootstrap over episodes, 2000 draws; paired differences.
- n per horizon is printed (only 7 of the research episodes reach 8Q), with the minimum detectable effect.
- Favourable and unfavourable paired intervals are counted against chance. In the research round: 1 of 30 favourable,
  3 unfavourable.

**Decomposition** (promoted to the headline):
- total = (N − O-full), the hypothesis and translation error, + (O-full − realized), the simulation error;
- (O-B − O-A′) is what the financial-conditions part of the hypothesis carries;
- for H-SEP: (H-SEP − O-A) is the hypothesis error of the Fed's forecast.

**Strata:** the clean set (headline); the short-training set; busts against ordinary episodes (pre-registered,
secondary).

**Claims** (pre-registered in `preregistered/<date>_TN.json`; the generated text may say nothing stronger):

| Id | Claim | Condition |
|---|---|---|
| C1 | 「叙事带来了改进」 | N2 − U equity9 window-end MAE upper bound < 0 on the clean set **and** N2 − N-wrong upper bound < 0; otherwise 「没有可检出的改进（最小可检出差异约 X）」 |
| C2 | 「金融条件承载了结果」 | O-B − O-A′ upper bound < 0 |
| C3 | 「季度模型比月度引擎好」 | Q − E upper bound < 0 on O-B and on N2 |
| C4 | recognition rate and contamination index | always shown next to C1–C3 |

### 8.7 How results are shown

The evaluation tab (§10.1 C) shows:
- a run selector covering the research, mock and real runs, with backend, model and date;
- the scoreboard (arms × horizons) with intervals;
- per-episode small multiples: realized against each arm's median and 80% band, quarterly;
- the blindness panel;
- the decomposition bar;
- discrimination (right against wrong narrative);
- the estimate and run buttons.

---

## 9. The forecast tab: 未来一年：情景条件模拟

The tab is renamed from 一年期预测. Its id stays `forecast`, so deep links still work.

| Section | Content |
|---|---|
| 1. 一句话 | 「这不是预测：下面是 13 个情景在未来 4 个季度的模拟，按情景概率加权。预测只在情景概率里。」 |
| 2. 情景加权的季度区间 (headline) | per asset, cumulative quarter-end fans for Q1–Q4, mixed by the probabilities; a table of p10 / p50 / p90 per quarter |
| 3. 每个情景的季度中位线 + 「你必须相信什么」 | per scenario, its stated ff, unemployment, CPI, oil, credit stance and valuation mode for the 4 quarters, with stated / completed badges |
| 4. 概率 | inputs for the 13 probabilities (sum to 1, reset to defaults), and each asset's sensitivity to each probability (mean_i − mixture mean) |
| 5. 无情景基准（历史常态） | the unconditional quarterly band as a reference line. The analog band is shown only where it differs; it is flat at 38 of 42 origins today, and the text says 「相似起点法与无条件抽样相当」 (the regime round's F0a/F0b) |
| 6. 情景组合隐含的宏观基率 | P(contraction within 4Q), P(CPI y/y > 4%), P(ff ≥ +1 pp) against history and, when available, the SEP |
| 7. 前瞻账本 | each build appends the mixture's quarterly quantiles to `data/results/ledger/forecast_ledger.jsonl` with model, card and probability versions. Entries are scored as quarters arrive (CRPS, PIT, coverage), with 「需要很多年才能分辨中等程度的偏差」. The existing rolling one-year record moves below as 历史回放（相似起点法） |

The engine is the monthly engine aggregated to quarter-ends until T-S passes, then the quarterly stack.

---

## 10. UI

### 10.1 New tab 情景条件检验 (`evaluation`, first in the 验证 group)

**A. 结论.** One paragraph generated from the pre-registered claims and the numbers.

**B. 已知因子路径时能复现多少.**
- Rows: O-A, O-A′, O-B, O-full; L0, G (if adopted), NN (report-only).
- Columns: 1Q and 4Q in full; 12Q and 20Q greyed with 「每个十年只有 2–3 个独立窗口，只能当个案」.
- Cells: R² [90% CI], CRPSS, 80% coverage, n_indep.
- Explanation card: the critic's ablation (pure macro, F without credit, GZ alone).

**C. 叙事检验.** As in §8.7.

**D. 估值账本.**
- Per window: realized price = multiple + fundamentals.
- Predicted against realized multiple, with the placebo row.
- Today's state table with percentiles and caveats.

**E. 因子引擎检验.**
- The T-F table: statistic, history, simulated p05 / p50 / p95, pass.
- Delivery charts.
- Fill-in skill.

**F. 前瞻账本.** A link to the forecast tab.

**G. 预注册与选择记录.**
- Every frozen record: date, hash, what was selected on which data, the result.
- Contamination notes: 2015–26 has been seen; F's 1995–2004 was its validation period; the schema tiers used all three
  decades.

**Monthly R² is demoted:**
- the diagnostics tab's tiles move into a card 月度传导（参考）;
- the backtest tab's headline tile becomes quarterly and annual;
- the overview and method texts are updated.

### 10.2 New tab 变量卡片 (`card`, in the 说明 group)

- A searchable table with the Appendix A columns plus today's value, its percentile and a sparkline.
- Filters by 槽位, 时点, 内生性 and 到资产的路径.
- Each row links to its evidence.
- The same component provides tooltips in the lab's QTable view.

### 10.3 Scenario detail

- **「情景说了什么 / 模型补了什么」:** a per-quarter table with badges 文字数字 / 文字定性 / 推算 / 模型补全 / 锚.
- **估值 block:**
  - the mode;
  - the implied CAPE and P/D paths (p10 / p50 / p90);
  - 「你必须相信的估值」, for example 「10 年后 CAPE 约 X（今天 41.2，历史第 97.8 百分位）」;
  - the valuation-statement wording of §6.3.
- **Attribution bars:** stated, completed, drift, valuation statement.
- **The financial-conditions block.**
- **The regime readout** (existing), applied to quarterly paths.

### 10.4 Lab (情景实验室)

- **Flow:** narrative → QTable → validation report → run → results.
  - The validation report lists refusals, warnings, and any missing R1–R5 with the evidence sentence.
  - The run uses the quarterly stack (试运行) or the monthly engine.
- **Editing:**
  - per-quarter cells (the user may fill any field);
  - a valuation-mode control and an fc-stance control;
  - the rarity statistic per quarter.
- **Monthly-engine runs** show which views the conversion dropped.

### 10.5 Other tabs

- **Overview and method:** 「模拟，不是预测」; quarterly horizons; the answer to the 4th message (§1); the NN line (§4.3).
- **Diagnostics:** monthly R² demoted (§10.1).
- **Backtest:**
  - anchors fitted to the train end (note);
  - fan_cond split into 「宏观 + 金融条件（含市场价格）」 (the current VIEW_VARS: s_cpi_yoy, s_ip_yoy, f_ff, f_y10,
    f_spread, f_oil, f_unrate, f_usd) and a new 「只有宏观」 (s_cpi_yoy, s_ip_yoy, f_ff, f_oil, f_unrate).
- **Valuation tab:**
  - CAPE from Shiller 2026;
  - the identity decomposition;
  - placebo-corrected statements;
  - the QQQ limitation;
  - the FCF caveat.

### 10.6 Wording rules (enforced in review and by a text test)

- 「预测」 appears only for Layer 1 (probabilities) and the ledger. Scenario outputs are 情景条件模拟 or 推演.
- Every number comes from JSON, and every stated or completed field is labelled.
- 12–20Q figures always carry the anecdote label.
- No improvement is claimed unless a pre-registered claim passes.
- 估值假设 always carries 「相当于直接假设股价」.

---

## 11. Interfaces to the agent tree, instruments, portfolio and withdrawal

### 11.1 Agent tree (`feat/agent-tree`, `scenariolab/agents/model_adapter.py`)

**Loading.** `scenariolab.qmodel.adapter:make_adapter`, loaded with
`MSL_AGENT_MODEL=scenariolab.qmodel.adapter:make_adapter`.

**What it provides** (the tree's documented `ModelAdapter` contract):
- `version`; `stub = False`.
- `capabilities = {"structural", "rarity", "impute"}`:
  - structural = the u_* shocks under the recursive identification of §5.3;
  - rarity = the §3.5 statistic;
  - impute = filling root fields not yet released at the origin (2026Q3 NIPA and Z.1) from the joint distribution.
- `x_fields`: the 52 schema ids. `assets` in `config.ASSET_IDS` order. Because `Given` holds floats, `x_oil_cause` is
  passed as 0 (demand), 0.5 (mixed) or 1 (supply), the qdata convention, and `field_info` documents it.
- `field_info()` with `hist` (p01 … p99, sd_q, sd_q_1985_2019).
- `root_state(n)` at the start of 2026Q4.
- `complete()`, `update_state()`, `rarity()`.

All REQUIRED_STATE keys are provided:
s_ff, s_unrate, s_core_yoy, s_spread, s_term, s_infexp1, s_rstar, s_sahm, s_cape, d_ff_l1, d_unrate_l1, d_unrate_l2,
d_spread_l1, d_itinv_l1, d_gz_l1, us_equity_l1, nasdaq_4q, real_oil_d1, real_oil_d2.

**Draws:**
- `y` = quarterly log total returns;
- `y["mult"]` = Δlog CAPE multiple. This changes the adapter docstring's Δlog P/Ê (request R-A2);
- `inc` = income; `cpi_q`;
- an optional `drawn` array of residual-quarter labels (request R-A4).

`validate_adapter()` must pass; this is a test in both repositories.

**Requests to the agent round:**

| Id | Request |
|---|---|
| R-A1 | rebuild the agents overlay on schema 3.0: `x_oil_cause` enum; `y_mult` redefined; `s_ecyc` and `y_eps` new; `s_pe` display-only |
| R-A2 | adopt the y["mult"] semantics above |
| R-A3 | price the first real snapshot (beam 2 × 80 quarters) with this adapter, not GaussianStub. The LLM's decisions depend on model-priced observation packets, so re-pricing a stub snapshot would miss its cache. Run it after phase P4 here (§14) |
| R-A4 | accept the optional `drawn` field in Draws |

### 11.2 Instruments (`feat/instruments`, `CoreView.from_quarterly`)

**What the quarterly stack provides:**
- total and income per asset (simple returns);
- the CPI path;
- y10 = s_y10 + Σ d_y10; ff; spread; d_gz; d_tp10;
- tb3 (ff plus the training-mean bill spread);
- `drawn` (residual-quarter labels, for pairing).

**T18** is restated as the F5 posterior predictive check. History: quarterly corr(d_gz, us_equity) = −0.58 and
corr(d_spread, us_equity) = −0.60 on 1985–2026 (n 167), or −0.56 and −0.40 on 1973–2026 [d01]. The instruments design's
targets (−0.65 ± 0.15, −0.64 ± 0.15) should name their sample. Request R-I1: restate them on 1985–2026.

### 11.3 Portfolio and withdrawal

Both get quarterly scenario sets (`step_months = 3`; volatility × √4; quarterly rebalancing; yearly withdrawals
unchanged). The agent tree's P11 needs the same change. Whichever round lands first implements it, and the other reuses
it. Until the switch, both services keep the monthly sets.

---

## 12. Interaction with the running rounds: landing order and merge

**What changed since the task was written.** The factor round has landed (`12b4141`, merged into `feat/causeway-v2` at
`773dfb7`). The regime layer has landed (`9146ca4`), and its follow-up is uncommitted in RWT. The agent-tree and
instruments rounds are running from `773dfb7`.

**Landing order.**
1. **The regime follow-up in RWT** commits to `feat/causeway-v2`. It touches `scenarios/regime_views.py`, `calibrate.py`,
   the spec YAMLs and `_scenario_detail.js`.
2. **P0 of this spec**, a small fix PR on `feat/causeway-v2`, right after step 1. It edits `analysis/backtest.py`,
   `llm/translator.py`, `scenarios/schema.py`, `data/panel.py` and the data import, which the follow-up also edits.
3. **`feat/instruments` and `feat/agent-tree`** land when their own gates pass. The agent tree lands on GaussianStub;
   real snapshot runs wait for step 4 (R-A3).
4. **P1–P4 of this spec** (new packages; additive; can be developed in parallel from step 2) land. Then the agent tree's
   P14 swaps in the real adapter, and instruments add `CoreView.from_quarterly`.
5. **P5:** the pre-registered T-A run, once, recorded.
6. **P6–P8** (QTable translator, narrative test, forecast tab, UI) after step 3, to avoid conflicts in `app.js` TABS,
   `method.js`, `_scenario_detail.js`, `server.py` router includes and `build.py` steps.
7. **P9, the switch gate T-S.** If it passes, the scenario, forecast, portfolio and withdrawal defaults switch.
8. **`main` ← `feat/causeway-v2`** when the user approves.

**Merge rules.**
- Develop in a new worktree `causeway-quarterly` on `feat/quarterly-model`, cut from `feat/causeway-v2` after step 1.
- Rebase onto the integration branch; one PR per phase; the suite green at every PR.
- Expected conflicts: `translator.py`, `scenarios/schema.py`, `server.py`, `build.py`, `web/js/app.js`, `method.js`,
  `_scenario_detail.js`, README, ARCHITECTURE. Resolve them by keeping both sides' additions.
- `config.py` is not edited; the quarterly stack's constants live in `qmodel/config_q.py`.

**How the landed rounds feed this spec:**
- **Factor round.**
  - L0 is its F (`model_selection.json`).
  - Its data fixes (spread timing, release-dated quarterly series, GZ, ACM TP, metals/oil) are the quarterly panel's
    inputs.
  - Its monthly walk-forward stays as the diagnostics reference.
  - Its D1 (MLP report-only) agrees with §4.3.
  - Caveats carried into the model card: GZ was selected with 1995–2004 as validation; ACM is re-estimated on the full
    sample.
- **Regime layer.**
  - Regime requests in QTables become inequality constraints using the classifier's rules.
  - The regime readout labels quarterly paths. The joint study labelled quarters with the regime classifier; its monthly
    implementation reproduces the engine reference exactly (40.77% / 17.29%).
  - The W-warnings are reused.
  - Its follow-up items stay open: F0b, built-in spec content, the CPI drift. The quarterly stack's F2 test covers the
    CPI drift (40.8% of months above 4% in the engine against 33.2% in history).

---

## 13. Decisions for the user (each with a default)

| # | 决定 | 默认 |
|---|---|---|
| U-1 | 默认引擎何时从月度引擎换成季度模型 | 通过预注册的切换关 T-S 才换（§4.6）；在那之前季度模型标“试运行”，只给情景实验室和智能体推演用 |
| U-2 | 金融条件（信用利差的方向）是否每个阶段必填 | 必填；不写时由模型补全，并标“补全的信用利差在历史检验里与无条件抽样相当” |
| U-3 | 情景没说估值时怎么办 | inferred：只有宏观与信用隐含的部分，残差倍数零漂移，不回归 |
| U-4 | 估值倍数的口径 | CAPE 倍数（CRSP 价格 /（CPI × 10 年平均实际 EPS））；另两档：派息调整的 P/D、原 P/Ê（不推荐：减记扭曲） |
| U-5 | 长期回报是否扣除历史估值抬升 | 扣除（训练期 CAPE 倍数每年抬升约 1.2–1.9%，不往未来外推）；另一档：保留训练期平均回报 |
| U-6 | 真实大模型的后端 | 本机 Claude Code（你的订阅额度，你已定）；API 只在你配置 ANTHROPIC_API_KEY 并设 MSL_LLM_BACKEND=api 时，按 token 计费 |
| U-7 | 叙事检验的片段 | 原 12 个 + 新增 1987–88、1998–2000、2010–12、2024–26（干净集 14 个）；1977–79、1983–84 进训练期短的一组；全部按盲测规范 v2 重写叙事（需要一轮写作） |
| U-8 | 真实调用的叙事检验 | 只在你点按钮或运行命令、看过估算（默认约 116 次调用）并确认后进行；构建与测试永远用模拟后端 |
| U-9 | 情景概率 | 用手设概率；按今天状态自动倾斜关闭，直到领先宏观状态通过预注册检验 |
| U-10 | 神经网络 | 只作对照（三个十年都输） |
| U-11 | 回测的长期锚 | 用训练期末可知的规则（会改变现有回测数字） |
| U-12 | 极端季度（如 2020Q2 那样的冲击）在模拟里出现的频率 | 按历史频率（约每 214 个季度一次，10 年路径里约 17% 含一次），估计时用不按日期的稳健权重；另一档：情景层明确调低 |
| U-13 | 12–20 个季度的统计 | 显示但灰化，标“每个十年只有 2–3 个独立窗口，只能当个案”，不进标题 |
| U-14 | QQQ 的估值 | 用全市场 CAPE 近似，BusEq 市净率只作显示；允许的数据源里没有纳斯达克 100 的市盈率或自由现金流 |
| U-15 | 用“美联储当时的预测（SEP）”做事前检验 | 做（从 ALFRED 取实时版本，不用大模型），排在叙事检验之后 |
| U-16 | 情景长度 | 60 个季度（15 年，与组合和提取一致），主要看 40 个季度（10 年） |

---

## 14. Implementation order, tests and docs

Each phase ends with its tests green, and nothing later depends on an unfinished earlier phase. In every test the LLM is
the mock.

| Phase | Build | Exit tests |
|---|---|---|
| **P0** (fixes to today's app; small PR) | **P0.1** `engine/anchors.py`: `train_end_anchors(panel, end)` (§5.5); `backtest.run_window` passes `reversion=`; backtest.json re-run. **P0.2** fan_cond split into 宏观+金融条件 and 只有宏观. **P0.3** s_cape from the Shiller 2026 workbook (import as the factor round imports its raw files). **P0.4** translator: endogenous-view warnings extended; VALUATION_VARS refused in test mode. **P0.5** `schema.json` → project `scenariolab/qmodel/schema.json` with the `x_oil_cause` enum and the `y_mult` redefinition; `tools/schema_view.py` (blind view). **P0.6** (optional) a per-phase `term_premium` override in the monthly engine, so a stated policy path no longer forces the 10-year target | anchors use rows ≤ end only (truncation test: building at end t equals the full build's value at t); DEFAULT_REVERSION still used forward; the new fan_cond arm reproduces the old one when VIEW_VARS are equal; s_cape equals Shiller's CAPE to 1e-6 where both exist; the blind view contains no 19xx/20xx token and no `footprint_quarters`; the existing suite unchanged elsewhere |
| **P1** quarterly data and the card | `qmodel/panel.py` (port qdata.py + valblock.py); `card.py` + `tools/build_model_card.py`; `panel_quarterly.csv` with meta | a quarter's change equals the window sum of the monthly factor changes (≤ 1e-12); `__known` columns respect release lags (truncation test); V-PRICE identity Δlog P = y_mult + y_fund to 1e-10; card ↔ panel ↔ QTable schema one-to-one; reaction functions reproduce u_ff R² 0.54 / 0.51 / 0.46 |
| **P2** asset layer, valuation, harness | `assets.py` (L0, residual model, income, mechanical assets); `valuation.py`; `evaluation/harness.py` (port of simharness with placebo anatomy) | L0 reproduces the qmodel's L0 1Q scores (0.126 / 0.261 / 0.441) to 1e-3; drift-only R² = 0 to 1e-12; simharness `selfcheck()` values reproduced; placebo row present; T-A candidate flags off by default |
| **P3** factor layer | `factors.py` (BVAR core, satellites, EM, Student-t weights, WZ conditioning, do-operator, shadow rate, lu transform, cycle module, completion shrinkage, regime covariance as a flag); `policy.py` | OLS limit of the BVAR to 1e-6; hard constraints to 1e-8 on every path; Gaussian MC mean = analytic conditional mean (\|z\| < 3); a stated u_ff leaves same-quarter E0 unchanged exactly; ff ≥ 0.1 always; determinism by seed; T-F F1–F10 and T-C computed and stored in `qmodel.json` (pass/fail shown; no gate yet) |
| **P4** rollout and adapters | `rollout.py` (path, step, continuation, state roll, disaggregation for export), `adapter.py`, `api.py` (`/api/q/*`) | `validate_adapter()` of the agent tree passes; step/path consistency (KS); state-roll identities; T18 / F5 statistics; performance: step mode N = 500 ≤ 1 s, path mode 40Q × 1,000 paths ≤ 10 s; **the agent tree's P14 can start** |
| **P5** pre-registered run | freeze `preregistered/<date>_TA.json`, `_TC.json` and `_TS.json` (protocol, candidates, thresholds, seeds, data/card/code hashes); run T-A and T-C once | the record exists before the run (commit order checked in CI); results appended, never edited; the default asset layer set by the T-A rule; the NN row recorded |
| **P6** QTable translator and scenario layer | `qtable.py` (validate, constraints, `to_phase_spec`); the QTable prompt (production and test modes); the mock backend; validation messages; `tools/qtable_from_yaml.py` and the 13 reviewed QTables | the JSON schema validates the examples; every refusal rule has a fixture; mock end-to-end: narrative → QTable → run → report; malformed JSON, refusal, timeout and usage-limit fixtures map to repair / fallback / pause; YAML → QTable delivery test; test-mode prompt scans clean |
| **P7** narrative test | episode registry + sealed key; `prompts.py` (blind), `runner.py`, `score.py`, `audit.py`, `report.py`; `import-standin`; estimate/run commands and endpoints; `sep.py` (ALFRED SEP series; download only with the user's approval, into its own folder) | the prompt builder cannot import the key; the mock run passes end to end; statistics recomputed from the scorer's stored rows (`score_rows.csv`) equal its published pooled numbers on the original 11 episodes (equity9 MAE ± 0.002) and the critic's subsets; `/run` refuses without a fresh estimate id and `confirm: true`; pytest asserts the mock backend |
| **P8** UI | the evaluation and card tabs; the forecast tab rework (§9); scenario detail (stated/completed, valuation, attribution); lab QTable editor; method, overview, diagnostics, backtest and valuation texts | renders on fixtures at 1,100 px and 375 px; no LLM text reaches `innerHTML`; the wording test (no 「预测」 in scenario-result titles; anecdote labels on 12–20Q); deep links unchanged |
| **P9** switch | T-S evaluated from P5 and P7 records; if it passes: defaults switch, quarterly sets for portfolio and withdrawal (shared with agent-tree P11), the 13 scenarios rebuilt on the quarterly stack | the existing portfolio and withdrawal tests unchanged with monthly sets; new tests with `step_months = 3`; if T-S fails, nothing switches and the reasons render |
| **P10** docs | ARCHITECTURE.md chapter 季度模型 (the three layers, the cell, the card, the QTable, the factor layer, valuation, evaluation); README (how to run evaluation and the narrative test; real runs only by hand); CONTRIBUTING (tests never spend tokens; never edit a frozen pre-registration record); `docs/research/2026-11-02-sim-redesign/` (the ten reports, the critique, this spec, d01–d05) | doc links checked; the README's commands run on the mock |

**Docs to update**, besides P10: the method tab's step list (a quarterly chapter replaces 「10. 估值是状态，不是因子」's
monthly framing); the in-app 计算口径与局限 cards; `.env.example` (`MSL_LLM_BACKEND` also accepts `mock`, for tests
only).

---

## 15. Open questions

1. **Is 2015–26 acceptable as the pre-registered test decade?** Every earlier round has seen it. The alternative is to
   wait years for the prospective ledger, the only clean holdout.
2. **Does a stated credit-spread path still count as "scenario → factor → asset"?** It is close to stating the equity
   risk premium (critic 1). This spec says yes, with the label, but it is the user's call (U-2).
3. **The 2026H1 jump in Shiller E** (+23% in two quarters): reported or estimated? Today's P/E, E10 and s_ecyc depend on
   it.
4. **Real-time vintages.** ALFRED re-estimation of the reaction functions, the stabilizers and the NIPA/Z.1 states has
   not been done. The CBO NROU and HLW real-time series could replace the crude anchor rules if their vintages reach
   back far enough.
5. **The unemployment floor:** the log(u − 2) transform against a logit on (2, 20), decided in T-F (F6).
6. **QE magnitudes** (template G08) can come only from the 2008–14 footprints: 26 quarters, 4 programs.
7. **Agent round requests R-A1–R-A4.** Is the y["mult"] change acceptable, and will the first real snapshot wait for P4?
8. **Instruments' T18 sample** (R-I1): which sample defines the target?
9. **Retiring the monthly engine.** After a successful switch, should it be retired, or kept indefinitely as a reference
   arm?
10. **Short-training episodes** (E03, E06, N16, N17). Report them apart (default) or drop them?
11. **Tail frequency.** A 2020Q2-size quarter appears in about 17% of 10-year paths at history's frequency (U-12).
    Should the default scenario layer lower it?
12. **Backend instruction conflict.** The task text asked how the test runs "with the real Anthropic API once a key is
    configured". USER_DECISIONS.md records 「Do not use the Anthropic API」 and the local Claude Code CLI. This spec
    supports both behind one interface, with claude_code as the real default and api only when explicitly configured.
    The lead or the user should confirm.
13. **The bust stratum.** It is defined on the outcome (a drawdown ≤ −20%), which is acceptable only as a pre-registered
    secondary stratum. Is it acceptable?
14. **Contemporaneous Fed text** (FOMC statements, federalreserve.gov) as c_q for the H-SEP arm: worth the LLM calls, or
    keep the SEP arm numeric only?
15. **The C′ cycle module is a hybrid** (non-parametric IP path plus the BVAR's co-movement). If its alternative
    (8-quarter innovation blocks) passes T-F equally, should the simpler one be preferred?

---

## Appendix A. The model card (generated by `design/d03_model_card.py` from schema-revise-2 3.0 plus this spec's overrides)

Columns:
- 同期/前定: the user's label;
- 到资产的路径: L0 = the factor round's F in the asset layer; 候选 = decided by T-A; 只经由因子层 = reaches assets only
  through its joint distribution with L0 fields;
- 翻译器: what the LLM may write;
- 金融条件: R5.

Rows are ordered x_q (52 fields + 5 shocks), then s_q (22), c_q anchors (5), outputs (5) and views (12). The CSV version
is `design/work/model_card.csv`.

| 字段 | 中文名 | 槽位 | 同期/前定 | 时点 | 内生性 | 到资产的路径 | 翻译器 | 核心因子层 | 金融条件 | 单位 | 起始 | 发布滞后 | 注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d_bdlev | 券商（非银）杠杆变动 | x_q | 同期 | 同季·实现 | E1 金融市场 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 只在文字明说时（数字或定性幅度） |  | 金融条件（可选） | log%/季 | 1973Q1 | Z.1 约 10 周 |  |
| d_bs | 央行资产负债表变动（QE/QT） | x_q（政策，经冲击 u_bs） | 同期 | 同季·实现 | E0p 政策（系统+冲击） | 只经由因子层 | 写原始值（因经济而动）或写冲击 u（干预） |  |  | pp GDP/季 | 1973Q1 | 0 | QE 对资产学不出（26 季、4 个计划）：资产效应只经由 d_tp10、d_y2、d_gz 的干预路径 |
| d_bund | 全球利率（德国 10 年期变动） | x_q | 同期 | 同季·市场/调查预期 | E1 金融市场 | 只经由因子层 | 只在文字明说时（数字或定性幅度） |  | 金融条件（可选） | pp/季 | 1973Q1 | 0；季均口径（月均） |  |
| d_corptax | 企业有效税率变动 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 可写 |  |  | pp/季 | 1973Q1 | 约 2 个月 |  |
| d_cpbill | 融资市场压力（商业票据 − 国库券利差变动） | x_q | 同期 | 同季·市场/调查预期 | E1 金融市场 | 只经由因子层 | 只在文字明说时（数字或定性幅度） |  | 金融条件（可选） | pp/季 | 1973Q1 | 0 |  |
| d_defense | 国防支出变动 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | pp GDP/季 | 1973Q1 | 1 个月 |  |
| d_deficit | 财政脉冲（联邦赤字/GDP 变动） | x_q | 同期 | 同季·实现 | E0p 政策（系统+冲击） | 只经由因子层 | 写原始值（因经济而动）或写冲击 u（干预） |  |  | pp GDP/季 | 1973Q1 | 约 2 个月 |  |
| d_epu | 经济政策不确定性变动 | x_q | 同期 | 同季·市场/调查预期 | E2 调查/新闻 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | log%/季 | 1973Q1 | 1 个月 |  |
| d_epu_health | 医疗政策不确定性变动（医疗行业） | x_q | 同期 | 同季·市场/调查预期 | E2 调查/新闻 | 只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | log%/季 | 1985Q2 | 1 个月 |  |
| d_ff | 政策利率变动（联邦基金利率） | x_q | 同期 | 同季·实现 | E0p 政策（系统+冲击） | L0 资产方程（因子轮 F） | 写原始值（因经济而动）或写冲击 u（干预） | 是 |  | pp/季 | 1973Q1 | 0 |  |
| d_gpr | 地缘政治风险变动 | x_q | 同期 | 同季·市场/调查预期 | E2 调查/新闻 | 只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | log%/季 | 1973Q1 | 1 个月 |  |
| d_hpi | 房价增长 | x_q | 同期 | 同季·实现 | E1 金融市场 | 只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | %年化 | 1975Q2 | 约 2 个月 |  |
| d_igrea | 全球实际活动（外需） | x_q | 同期 | 同季·实现 | E1c 商品价格 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | 指数/季 | 1973Q1 | 约 1 个月 |  |
| d_infexp1 | 短期通胀预期变动（密歇根 1 年） | x_q | 同期 | 同季·市场/调查预期 | E2 调查/新闻 | 只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | pp/季 | 1978Q2 | 0（调查月） |  |
| d_infexp_lr | 长期通胀预期变动（去锚/再锚的新闻） | x_q | 同期 | 同季·市场/调查预期 | E2 调查/新闻 | 只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | pp/季 | 1990Q3 | 0 |  |
| d_itinv | IT 设备与软件实际投资增长（通用技术资本开支） | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 1 个月 |  |
| d_lf | 劳动力增长（含人口与移民） | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 1 周 |  |
| d_lprod | 劳动生产率增长（非农） | x_q | 同期 | 同季·实现 | E0 外生宏观 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 5 周 |  |
| d_lshare | 劳动收入占比变动 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | log%/季 | 1973Q1 | 约 5 周（FWT 按发布月滞后 4 个月） |  |
| d_metals_oil | 基础金属/油价（区分油价的供给与需求） | x_q | 同期 | 同季·实现 | E1c 商品价格 | L0 专用（能源） | 只在文字明说时（数字或定性幅度） |  |  | log%/季 | 1973Q1 | 约 1 个月 |  |
| d_natgas | 天然气价格 | x_q | 同期 | 同季·实现 | E1c 商品价格 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | log%/季 | 1973Q1 | 1 个月 |  |
| d_perstax | 个人所得税负担变动 | x_q | 同期 | 同季·实现 | E0p 政策（系统+冲击） | 只经由因子层 | 写原始值（因经济而动）或写冲击 u（干预） |  |  | pp/季 | 1973Q1 | 1 个月 |  |
| d_sent | 消费者信心变动（密歇根） | x_q（只在文字明说时） | 同期 | 同季·市场/调查预期 | E2 调查/新闻 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | 指数点/季 | 1973Q1 | 0（月中初值） | E2：同季资产收益解释其变化 adj R² 0.24；不许翻译器自选幅度 |
| d_tariff | 有效关税率变动 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | pp/季 | 1973Q1 | 1–3 个月 |  |
| d_transfers | 转移支付变动 | x_q | 同期 | 同季·实现 | E0p 政策（系统+冲击） | 只经由因子层 | 写原始值（因经济而动）或写冲击 u（干预） |  |  | pp/季 | 1973Q1 | 1 个月 |  |
| d_usd_em | 美元对新兴市场货币（EM 压力） | x_q | 同期 | 同季·市场/调查预期 | E1 金融市场 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 只在文字明说时（数字或定性幅度） |  | 金融条件（可选） | log%/季 | 1973Q2 | 0 |  |
| x_lending_std | 银行放贷标准（信贷供给） | x_q | 同期（描述上一季，季中发布） | 滞后一季 | E2 调查/新闻 | 只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | % | 1990Q2 | FRED 把 4 月调查记在第 2 季，问的是过去 3  |  |
| x_oil_cause | 油价变动的原因（需求 / 供给） | x_q | 同期 | 同季·实现 | E0 外生宏观 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 可写 |  |  | 0/1 | 1973Q2 | 0（叙事）；训练时同 d_igrea（约 1 个月） | 枚举 supply / demand / mixed / null（不再用 0/1：翻译器与 qdata 的约定相反） |
| x_supply_chain | 供应链压力 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 可写 |  |  | 扩散指数 | 1973Q1 | 0（调查月） |  |
| d_capex | 实际企业投资增长 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 1 个月 |  |
| d_chargeoff | 银行贷款核销率变动（信用损失） | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | pp/季 | 1985Q1 | 约 2 个月 |  |
| d_cons_goods | 实际商品消费增长 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 1 个月 |  |
| d_cons_serv | 实际服务消费增长 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 1 个月 |  |
| d_core | 核心通胀 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 2 周 |  |
| d_cpi_goods | 商品通胀 | x_q | 同期 | 同季·实现 | E0 外生宏观 | L0 资产方程（因子轮 F） | 可写 | 是 |  | %年化 | 1973Q1 | 约 2 周 |  |
| d_cpi_serv | 服务通胀 | x_q | 同期 | 同季·实现 | E0 外生宏观 | L0 资产方程（因子轮 F） | 可写 | 是 |  | %年化 | 1973Q1 | 约 2 周 |  |
| d_cre | 商业地产价格变动 | x_q | 同期 | 同季·实现 | E1 金融市场 | 只经由因子层 | 只在文字明说时（数字或定性幅度） |  |  | log%/季 | 1973Q1 | Z.1 约 10 周 |  |
| d_exports | 实际出口增长 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 1 个月 |  |
| d_gz | GZ 信用利差变动 | x_q | 同期 | 同季·市场/调查预期 | E1 金融市场 | L0 资产方程（因子轮 F） | 只在文字明说时（数字或定性幅度） | 是 | 金融条件（必填其一） | pp/季 | 1973Q2 | 0 | 因子轮以 1995–2004 为验证期选入（该十年不是干净的检验）；同季资产收益解释其变化 adj R² 0.40：它是市场价格，写它接近写股权风险溢价 |
| d_ip | 工业产出增长 | x_q | 同期 | 同季·实现 | E0 外生宏观 | L0 资产方程（因子轮 F） | 可写 | 是 |  | log%/季 | 1973Q1 | 约 2 周 |  |
| d_oil | 油价变动 | x_q | 同期 | 同季·实现 | E1c 商品价格 | L0 资产方程（因子轮 F） | 只在文字明说时（数字或定性幅度） | 是 |  | log%/季 | 1973Q1 | 0 |  |
| d_profits | 企业税后账面利润增长（NIPA CP） | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | log%/季 | 1973Q1 | 约 2 个月（第二次 GDP 估计） |  |
| d_resinv | 实际住宅投资增长 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 1 个月 |  |
| d_rgdp | 实际 GDP 增长 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 1 个月（初值） |  |
| d_spread | Baa − 10 年利差变动（计时修正后） | x_q | 同期 | 同季·市场/调查预期 | E1 金融市场 | L0 资产方程（因子轮 F） | 只在文字明说时（数字或定性幅度） | 是 | 金融条件（必填其一） | pp/季 | 1973Q1 | 0 | 1962–1985 的计时已修正（因子轮）；同季资产收益解释其变化 adj R² 0.32 |
| d_tips10 | 10 年 TIPS 实际收益率变动（黄金） | x_q | 同期 | 同季·市场/调查预期 | E1 金融市场 | 候选（T-A 预注册检验决定）；之前只经由因子层 | 只在文字明说时（数字或定性幅度） |  | 金融条件（可选） | pp/季 | 2003Q1 | 0 |  |
| d_tp10 | 10 年期限溢价变动（ACM） | x_q | 同期 | 同季·市场/调查预期 | E1 金融市场 | L0 资产方程（因子轮 F）：只作 TP×1{核心>4} | 只在文字明说时（数字或定性幅度） |  | 金融条件（可选） | pp/季 | 1973Q1 | 0 | ACM 用全样本参数估计（轻度前视）；在 F 里只以 TP×1{核心>4} 进入 |
| d_unrate | 失业率变动 | x_q | 同期 | 同季·实现 | E0 外生宏观 | L0 资产方程（因子轮 F） | 可写 | 是 |  | pp/季 | 1973Q1 | 约 1 周 |  |
| d_usd | 美元变动 | x_q | 同期 | 同季·市场/调查预期 | E1 金融市场 | L0 资产方程（因子轮 F） | 只在文字明说时（数字或定性幅度） | 是 | 金融条件（可选） | log%/季 | 1973Q2 | 0 |  |
| d_wage | 工资增长 | x_q | 同期 | 同季·实现 | E0 外生宏观 | 只经由因子层 | 可写 |  |  | %年化 | 1973Q1 | 约 1 周 |  |
| d_y10 | 10 年期收益率变动 | x_q | 同期 | 同季·市场/调查预期 | E1 金融市场 | L0 资产方程（因子轮 F） | 只在文字明说时（数字或定性幅度） | 是 | 金融条件（可选） | pp/季 | 1973Q1 | 0 |  |
| d_y2 | 2 年期收益率变动（预期政策路径） | x_q | 同期 | 同季·市场/调查预期 | E1 金融市场 | 只经由因子层 | 只在文字明说时（数字或定性幅度） |  | 金融条件（可选） | pp/季 | 1976Q3 | 0 |  |
| u_ff | 政策利率冲击 | x_q（干预） | 同期 | 同季·实现 | E0p 冲击 | 经由因子层（递归识别：同季只动 E1/E2，宏观滞后反应） | 只在文字写明“干预”时（数字或定性幅度） |  |  | pp/季 | 1973Q1 | — | 系统部分由反应函数/稳定器在训练期末前拟合 |
| u_bs | 资产负债表冲击 | x_q（干预） | 同期 | 同季·实现 | E0p 冲击 | 经由因子层（递归识别：同季只动 E1/E2，宏观滞后反应） | 只在文字写明“干预”时（数字或定性幅度） |  |  | pp GDP/季 | 2008Q3 | — | 系统部分由反应函数/稳定器在训练期末前拟合 |
| u_def | 赤字冲击 | x_q（干预） | 同期 | 同季·实现 | E0p 冲击 | 经由因子层（递归识别：同季只动 E1/E2，宏观滞后反应） | 只在文字写明“干预”时（数字或定性幅度） |  |  | pp GDP/季 | 1973Q1 | — | 系统部分由反应函数/稳定器在训练期末前拟合 |
| u_tr | 转移支付冲击 | x_q（干预） | 同期 | 同季·实现 | E0p 冲击 | 经由因子层（递归识别：同季只动 E1/E2，宏观滞后反应） | 只在文字写明“干预”时（数字或定性幅度） |  |  | pp/季 | 1973Q1 | — | 系统部分由反应函数/稳定器在训练期末前拟合 |
| u_ptax | 个人税冲击 | x_q（干预） | 同期 | 同季·实现 | E0p 冲击 | 经由因子层（递归识别：同季只动 E1/E2，宏观滞后反应） | 只在文字写明“干预”时（数字或定性幅度） |  |  | pp/季 | 1973Q1 | — | 系统部分由反应函数/稳定器在训练期末前拟合 |
| s_cape | 起点估值：log 席勒 CAPE | s_q | 前定（期初已知） | 期初（前定） | 状态 | 恒等式 | 不可写（数据）；反事实起点只在实验室 |  |  | log | 1973Q1 | E 滞后约 3 个月 | 改用 Shiller 2026 版工作簿（E、D 到 2026-06）；MAIN 拼接版 2024–26 高 4–12.5%，要修 |
| s_fcfy | 自由现金流收益率（非金融公司） | s_q（只展示） | 前定（期初已知） | 期初（前定） | 状态 | 因子层的条件状态 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1953Q1 | NIPA 与 Z.1 按发布月：期初已知 q-2 | NIPA/Z.1 全部非金融公司的代理，不是 SPY/QQQ 的自由现金流收益率；水平状态样本外无增益 |
| s_infexp_lr | 长期通胀预期（锚定程度） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 锚 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1990Q2 | 0（调查月） |  |
| s_nfc_debt | 非金融企业债务/GDP | s_q | 前定（期初已知） | 期初（前定） | 状态 | 恒等式 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1973Q1 | Z.1：期初已知 q-2 |  |
| s_oil_real | 实际油价水平（log WTI/CPI） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 锚 | 不可写（数据）；反事实起点只在实验室 |  |  | log | 1973Q1 | CPI 滞后约 2 周 |  |
| s_payout | 总净派现收益率（股息 + 回购 − 新发行） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 恒等式 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1973Q1 | Z.1：期初已知 q-2 |  |
| s_pd | log 市价/股息（CRSP；各行业自身 P/D） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 恒等式 | 不可写（数据）；反事实起点只在实验室 |  |  | log | 1973Q1 | 0（月末价格与股息） |  |
| s_pe | 起点倍数：log P/Ê（CRSP 季末价格 / 4 季平均的滚动 12 个月 EPS） | s_q（只展示） | 前定（期初已知） | 期初（前定） | 状态 | 恒等式 | 不可写（数据）；反事实起点只在实验室 |  |  | log | 1973Q1 | Shiller E 滞后约 3 个月：期初已知的是 q− | 滚动 P/E 被 2001–02、2008–09 的减记扭曲：只展示，不进模型 |
| s_policy_gap | 政策立场缺口（实际政策利率 − r*） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 锚 | 不可写（数据）；反事实起点只在实验室 |  |  | pp | 1978Q1 | 同组成部分 |  |
| s_pshare | 税后企业账面利润/GDP 水平 | s_q | 前定（期初已知） | 期初（前定） | 状态 | 恒等式 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1973Q1 | 期初已知 q-2 |  |
| s_real_y10 | 10 年实际收益率 | s_q | 前定（期初已知） | 期初（前定） | 状态 | 因子层的条件状态 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1973Q1 | 0 |  |
| s_rstar | 中性实际利率 r*（HLW 单边估计） | s_q（锚） | 前定（期初已知） | 期初（前定） | 状态 | 锚 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1973Q1 | 回测必须用实时估计（当前版本只允许作同季 oracle  | 回测：训练期末可知的规则（过去 40 季实际政策利率均值）；HLW 当前版本只展示 |
| s_trend_g | 趋势增长（HLW g） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 锚 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1973Q1 | 期初已知 q-2 |  |
| s_usd_real | 实际美元水平（广义实际汇率） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 锚 | 不可写（数据）；反事实起点只在实验室 |  |  | log | 1973Q1 | 约 1 个月 |  |
| s_w | 行业市值权重（指数构成） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 恒等式 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1926Q3 | 0（月末价格） |  |
| s_core_yoy | 核心 CPI 同比（派生：高通胀体制 s_infl_regime = 核心 > 4%） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 因子层的条件状态 | 不可写（数据）；反事实起点只在实验室 |  |  | % / 0-1 | 1973Q1 | 1 个月 |  |
| s_ecyc | 盈利周期位置 log(Ê/E10) | s_q | 前定（期初已知） | 期初（前定） | 状态 | 锚 | 不可写（数据）；反事实起点只在实验室 |  |  | log | 1947Q1 | Shiller E 滞后约 3 个月：期初已知的是 q− |  |
| s_ff | 政策利率水平（联邦基金利率） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 因子层的条件状态 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1973Q1 | 0 |  |
| s_infexp1 | 短期通胀预期水平（密歇根 1 年） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 因子层的条件状态 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1978Q1 | 0 |  |
| s_spread | 信用利差水平（Baa − 10 年） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 因子层的条件状态 | 不可写（数据）；反事实起点只在实验室 |  |  | pp | 1973Q1 | 0（观测值，不修订） |  |
| s_term | 期限利差（10 年 − 政策利率） | s_q | 前定（期初已知） | 期初（前定） | 状态 | 因子层的条件状态 | 不可写（数据）；反事实起点只在实验室 |  |  | pp | 1973Q1 | 0 |  |
| s_unrate | 失业率 | s_q | 前定（期初已知） | 期初（前定） | 状态 | 因子层的条件状态 | 不可写（数据）；反事实起点只在实验室 |  |  | % | 1973Q1 | 1 个月 |  |
| a_oil | 实际油价锚移位 | c_q 锚 | 假设（情景给） | 情景路径（假设） | 情景参数 | 锚（情景层给，默认可知值） | 可写（有默认值） |  |  | log | 1973 | — |  |
| a_spread | 信用利差锚 | c_q 锚 | 假设（情景给） | 情景路径（假设） | 情景参数 | 锚（情景层给，默认可知值） | 可写（有默认值） |  |  | pp | 1973 | — |  |
| a_tp | 期限溢价锚 | c_q 锚 | 假设（情景给） | 情景路径（假设） | 情景参数 | 锚（情景层给，默认可知值） | 可写（有默认值） |  |  | pp | 1973 | — |  |
| a_usd | 实际美元锚移位 | c_q 锚 | 假设（情景给） | 情景路径（假设） | 情景参数 | 锚（情景层给，默认可知值） | 可写（有默认值） |  |  | log | 1973 | — |  |
| a_ustar | 失业率锚 | c_q 锚 | 假设（情景给） | 情景路径（假设） | 情景参数 | 锚（情景层给，默认可知值） | 可写（有默认值） |  |  | % | 1973 | — |  |
| y_assets | 13 个资产的季度对数回报（联合分布） | 输出 | 同期 | 同季·实现 | 输出 | — | 不可写（评估时掩码；情景估值只能写 c_q.valuation 的定性模式） |  |  | log | 1973Q1 | 0 |  |
| y_eps | 滚动 EPS 变化（含减记，只展示） | 输出（只展示） | 同期 | 同季·实现 | 输出 | — | 不可写（评估时掩码；情景估值只能写 c_q.valuation 的定性模式） |  |  | log | 1947Q1 | Shiller E 滞后约 3 个月 | 滚动 EPS 的变化：含减记，只展示，不作恒等式的腿；恒等式的盈利腿改为 y_fund = Δlog CPI + Δlog E10 实际 |
| y_mult | 倍数变化：Δlog CAPE 倍数（美股，CRSP 价格口径） | 输出（掩码） | 同期 | 同季·实现 | 输出 | — | 不可写（评估时掩码；情景估值只能写 c_q.valuation 的定性模式） |  |  | log | 1973Q1 | 0 | Δlog CAPE 倍数：CRSP 季末价格 /（CPI × 标普 10 年平均实际 EPS），valuation 组 us_equity_lm_cape；取代 schema 的 Δlog P/Ê（减记扭曲：2007Q3–2009Q2 P/Ê +1.005 而价格 −0.468） |
| s_bank_dur | 银行久期错配损失（视图） | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | pp | 1973Q2 | H.8 约 1 周 |  |
| s_debt | 联邦公众持有债务/GDP | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | % | 1973Q1 | 期初已知 q-2 |  |
| s_foreign_hold | 外国持有美债占比 | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | % | 1973Q1 | 期初已知 q-1 至 q-2 |  |
| s_gold_real | 实际金价水平（视图） | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | log | 1973Q1 | CPI 滞后约 2 周 |  |
| s_hh_debt | 家庭债务/可支配收入 | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | % | 1973Q1 | Z.1：期初已知 q-2 |  |
| s_house_pr | 房价/租金（视图） | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | log | 1981Q2 | 1 季 |  |
| s_int_burden | 联邦公众持有债务的利息负担（恒等式） | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | % | 1973Q1 | 期初已知 q-2 |  |
| s_it_share | IT 设备与软件投资/GDP | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | % | 1973Q1 | 期初已知 q-2 |  |
| s_lshare | 劳动收入占比水平 | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | 指数 | 1973Q1 | 约 5 周后发布：期初已知 q-2（FWT 按发布月滞后 |  |
| s_old65 | 65 岁以上人口占比 | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | % | 1960 | 年度，滞后约 1 年 |  |
| s_tdsp | 家庭偿债负担（视图） | 视图 | 前定（期初已知） | 期初（前定） | 视图 | 不进模型 | 只读 |  |  | % | 1980Q1 | 约 1 季 |  |
| wc_gap | 白领与蓝领工资增速差 | 视图 | 同期 | 同季·实现 | 视图 | 不进模型 | 只读 |  |  | pp | 1973Q1 | 约 1 周 |  |
| y_fund | 美股基本面腿（CPI × 10 年平均实际 EPS） | 输出 | 同期 | 同季·实现 | 输出 | 恒等式：y_us_equity = y_mult + y_fund | 不可写 |  |  | log | 1973Q1 | E 滞后约 3 个月 | 平滑：主要是 CPI 传导（valuation：R² 0.56，系数 0.97） |
| y_pd | 美股 P/D 变化（收入腿用） | 输出 | 同期 | 同季·实现 | 输出 | 恒等式：收入 = D/P | 不可写 |  |  | log | 1973Q1 | 0 | 约 30% 的 1973–2025 P/D 上升是派息率下降，不当作估值 |

## Appendix B. QTable/1 (the LLM's output = the model's input)

The full JSON Schema (draft 2020-12) is `design/work/qtable.schema.json`. It is generated by `design/d04_qtable_schema.py`
from `schema.json`, so its x-field enum is exactly the card's 52 x_q fields. Its parts:

| Part | Content |
|---|---|
| `schema`, `model_card`, `n_quarters` (≤ 60) | header |
| `rows[]` | `q`; `x` {field → entry}; `u` {u_ff, u_bs, u_def, u_tr, u_ptax → entry}; `fc` {stance (required): 明显收紧 / 收紧 / 中性 / 宽松 / 明显宽松 / 未说明, quote}. An absent field means not stated |
| entry | one of `value` (+ optional `sd`), `sign` + `size` (小 / 温和 / 明显 / 大幅 / 极端 = 0.25 / 0.5 / 1 / 2 / 3 training sd), `min` / `max`; always `src` ∈ {stated, stated_qual, derived} and `quote` (≤ 120 characters, a substring of the narrative) |
| `x.x_oil_cause` | `value` ∈ {supply, demand, mixed} |
| `c_q.anchors` | r_star, pi_lr, a_tp, a_spread, a_ustar, a_oil, a_usd: {value, from_q, quote} |
| `c_q.announced[]` | {field, quarters, values, sd, quote}: soft constraints on future quarters |
| `c_q.constraints[]` | {kind: sum / mean / yoy / level / range, field, quarters, value or min / max, sd, quote} |
| `c_q.valuation[]` | {quarters, mode: inferred / hold / rerate, direction, size: 温和 / 明显 / 极端, reversion_half_life_years ≥ 14, quote} |
| `c_q.regime[]` | {quarters, label, share, quote}: inequality constraints through the regime classifier |
| `events[]` | {text, quarters, templates [G..], channels, generic_footprint, how_decomposed} |
| `notes[]` | free text, never an input |

Example: a fragment of a four-quarter scenario. Each quote is assumed to appear in its narrative.

```json
{"schema": "qtable/1", "model_card": "card/1 (schema-revise-2 3.0 + SIM_DESIGN)", "n_quarters": 4,
 "rows": [
  {"q": 1,
   "x": {"d_oil": {"value": 26, "sd": 8, "src": "stated", "quote": "油价一个季度内上涨约三成"},
         "x_oil_cause": {"value": "supply", "src": "stated_qual", "quote": "产油国卷入战争，供应中断"},
         "d_gpr": {"sign": "+", "size": "大幅", "src": "stated_qual", "quote": "地缘政治风险急剧上升"},
         "d_cpi_goods": {"sign": "+", "size": "明显", "src": "stated_qual", "quote": "商品价格明显上涨"}},
   "fc": {"stance": "收紧", "quote": "信用利差走阔"}},
  {"q": 2, "x": {"d_ff": {"value": 0.5, "src": "stated", "quote": "央行加息 50 个基点"}}, "fc": {"stance": "未说明"}},
  {"q": 3, "x": {}, "u": {"u_ff": {"sign": "-", "size": "明显", "src": "stated_qual", "quote": "政治压力下央行意外降息"}},
   "fc": {"stance": "中性", "quote": "金融条件大体平稳"}},
  {"q": 4, "x": {}, "fc": {"stance": "未说明"}}],
 "c_q": {"anchors": {"r_star": {"value": 0.75, "quote": "中性利率维持在目前的估计"}},
         "constraints": [{"kind": "range", "field": "d_unrate", "quarters": [1, 2, 3, 4], "min": -0.2, "max": 0.3,
                          "quote": "就业基本稳定"}]},
 "events": [{"text": "产油国卷入战争", "quarters": [1, 2], "templates": ["G02"],
             "channels": ["d_oil", "x_oil_cause", "d_gpr", "d_epu"], "generic_footprint": true,
             "how_decomposed": "能源生产国卷入的战争 → 石油供给冲击、地缘风险上升、商品通胀"}]}
```

How the model reads it:
- **q1:** d_oil is a soft constraint at 26 ± 8 log% a quarter (+30% in the text is 26.2 log%). The supply label selects the oil-supply interaction. d_gpr
  and d_cpi_goods are soft constraints at 2 and 1 sd. The stance 收紧 is a soft constraint on d_gz and d_spread at
  +1 sd.
- **q2:** d_ff is hard at +0.5 and reads as a reduced-form observation (the reaction to inflation). The credit stance is
  completed, with warning W-FC.
- **q3:** a −1 sd intervention u_ff (the do-operator). Same-quarter E0 is not updated; markets react on impact.
- **q4:** no x field is stated, so everything is completed. Growth is pinned only by the unemployment range
  constraint, which applies in all four quarters; the cycle module therefore does not run in this example.
- **Valuation:** absent, so the mode is inferred.

## Appendix C. Critique findings → where this spec applies them

| Critique item | Applied in |
|---|---|
| 0.1 oracle-macro skill is credit-spread skill | §2 E1; §7.1 (R5 required); §8.3 (O-A against O-B); §8.6 claim C2; §10.1 B (ablation card) |
| 0.2 "misses are valuation swings" is an identity | §6.6 (corr(forecast, outcome), share predicted, placebo row); §10.1 D |
| 0.3 the narrative test was not blind | §8.4 (all ten channels and their checks); §8.2 (writer protocol v2); recognition probe; claim C4 |
| 0.4 "perfect translation would not help" overstated | §0 item 3 wording; §8.2 (more busts, short-training set apart); §8.6 strata |
| 0.5 DEFAULT_REVERSION look-ahead | §5.5; P0.1 (backtest tables re-run) |
| 0.6 wrong signs come from drift | §4.4 (attribution in every scenario detail; re-rating removed from forward drift) |
| 0.7 G − L0 is selection-sized | §4.2 (T-A: selection on 1990–2014, a single 2015–26 test, disclosure); L0 the default |
| 0.8 data errors | §6.5; P0.3 (s_cape); P0.5 and §3.4 (x_oil_cause enum); §6.1 (y_mult on CAPE); §4.4 (CAPE, not P/D, for re-rating) |
| 0.9 small samples | §4.5 (12–20Q coverage), §8.1 and §8.6 (MDE, n, counts against chance), §10.1 and §10.6 (anecdote labels) |
| 0.10 incomplete audit | §3.4 and §7.3 (a quote required for every value; E1/E2 without a stated basis refused); §8.4 (permanent audit) |
| 7.1–7.7 design consequences | 7.1 → §7.1, §8.3; 7.2 → §6.6; 7.3 → §8.4; 7.4 → §5.5, P0.1; 7.5 → §4.2, P5; 7.6 → §6.5, P0; 7.7 → §0, §8.6 |
| harness: F's 1995–2004 cells are validation, not test | §4.2 disclosure; §10.1 G |
| harness: e_*, EBP, ACM visible to plug-ins | §8.4 (no e_* or EBP inputs); model card note on ACM |
| harness: 12–20Q coverage indistinguishable from 0.80 | §4.5 |
| layers: R5 optional is a design flaw | §7.1 |
| layers: the ΔP/D "upper bound" is circular | §8.3 (arm V labelled 恒等式上界) |
| layers: the oracle arm feeds market prices | §8.3 (O-A / O-A′ / O-B split) |
| writer: unique events and sector/firm outcomes in narratives | §8.2 (protocol v2, banned-word scan) |
| writer: N2 states market prices; its gain sits in financials and the treasury | §4.1 and §8.3 (mechanical assets scored apart; N2 = the financial-conditions arm) |
| valuation: "by FCF not expensive" unsupported | §6.2 (display only, with caveat) |
| valuation: the multiple equation is mostly GZ | §6.2 and §6.6 (inference attributed to credit) |
| valuation: zero P/D drift ≠ zero re-rating | §4.4 (the CAPE multiple's drift) |
| valuation: the ECY interval is unreliable | §6.2 |
| joint: overlapping 13-year windows; bounce-back sample 1960Q2+ | §5.8 F2 and F4 definitions |
| joint: anchors are post-2008 constants | §5.5 |
| schema: λ_π is a one-episode check; PV duration imposes 3–5× the estimated rate sensitivity | V-PV and λ_π not used in v1; rate sensitivity only from estimated betas (§6.1) |
| schema: tiers selected on the test decades | §4.2 (re-selection on 1990–2014) |
| schema: schema.json as translator reference is a leak | §8.4 (blind schema view) |
| qmodel: level valuation states add nothing; Δy10 × P/D was scored on the test decade | §4.2 C2 candidate |
| qmodel: "stating both legs fixes it" is a tautology | §6.3 and §8.3 (valuation statements labelled as price assumptions; arm V) |
| qmodel: the 2020–21 check rests on about 2 extreme quarters | cited only for the sign of the generic channels under oracle x, not as proof that no flag is ever needed (§2 E12 scope) |
| qmodel: saturation fixed a priori | §3.5 |
| translator: r_star overrides in every spec | §7.3 (no per-phase r_star in QTable); P0.6 (term-premium override for the monthly engine) |
| translator: E1/E2 views with chosen magnitudes (16 f_sentiment, 3 f_profit_share) | §3.4 and §7.3 (quotes; refusal) |
| scorer: fan_cond is partly a market-price oracle | P0.2; §10.5 |
| scorer: "excluding E03 changes nothing" refuted | §8.2 (short-training set apart) |
| scorer: the engine oracle used B0, not F | §4.6 and §8.3 (engine arms run F) |

## Appendix D. Scripts of this role (read-only; run as `cd MAIN && .venv/bin/python -I -B <script> <inputs>`)

| Script | Output | What |
|---|---|---|
| `design/d01_history_stats.py` | `design/work/d01_history_stats.json` | quarterly correlations of credit and rates with us_equity; CAPE / P/D / P/Ê legs in busts; unemployment minimum; the CPI > 4% share |
| `design/d02_anchors_outliers_episodes.py` | `design/work/d02_anchors_outliers_episodes.json` | train-end anchors; the undated outlier screen (it flags much of 1973–82, hence Student-t weights rather than a cut-off); episode windows, training lengths, drawdowns, overlap check |
| `design/d03_model_card.py` | `design/work/model_card.md`, `model_card.csv` | Appendix A |
| `design/d04_qtable_schema.py` | `design/work/qtable.schema.json` | Appendix B's JSON schema |
| `design/d05_sizes_anchors.py` | `design/work/d05_sizes_anchors.json` | valuation re-rating size bins by span; CAPE-multiple drift by train end; the 80-quarter unemployment anchor |
