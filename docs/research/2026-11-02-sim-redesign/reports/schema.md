# The quarterly input table (schema-revise-2 3.0) (research round, role `schema`)

*What the role did:* the input table after two critique and revision rounds: policy split, templates, oracles.

Rendered from the role's final report in the research workflow's journal (run `wf_cc7fe8c3-afc`, result `schema`). The text is unchanged except that local absolute paths are written as the spec writes them (`scratchpad/sim_redesign/...`, `scratchpad/...`). It is the report the critic and the designer read; the numbers are the role's own, and the critique ([`critic/critique.md`](../critic/critique.md)) and the spec ([`SIM_DESIGN.md`](../SIM_DESIGN.md)) say which of them survived.

## Summary

schema-revise-2 rebuilt the scenario table from schema-revise-1 to address critic-2's 12 gaps. Decisions: 7 accepted, 1 accepted with a change, 4 partly accepted. Every rejected sub-part has a reason. Before writing, I re-tested the key claims with 6 read-only scripts (v01–v06) and made 1 FRED download of 2 requests (FYOIGDA188S; the key was never printed). Nothing in MAIN, RWT or FWT was written.

The answer to the user's question "what was missing, and why" is the opening section 0 of SCHEMA.md. It names three fundamental gaps.

1. Policy is an intervention, not a condition.
   - The five policy fields are now class E0p, each split into a systematic part plus a shock (do-operator): d_ff, d_bs, d_deficit, d_transfers, d_perstax.
   - Fit to train_end: the reaction function explains 40–54% of quarterly fed-funds changes; the automatic stabilizers explain 27–65% of deficit and transfer changes.
   - The raw fields imputed a recession for a cut or a stimulus. They also gave QE the wrong sign for equities.
   - QE cannot be learned from the data. Its asset effect now goes through intervention paths on d_tp10, d_y2 and d_gz.
2. The price = earnings × multiple accounting was not closed.
   - The multiple is now Δlog(P/Ê): CRSP quarter-end price over the 4-quarter mean of trailing EPS. The old CAPE multiple moved almost one-for-one with price.
   - New output y_eps carries the earnings leg, with an earnings bridge V-EARN from NIPA profits to index EPS.
   - New state s_ecyc = log(Ê/E10). It predicts both legs with opposite signs, but not the price.
   - Duration now uses the total net payout yield. Today's duration matches 1976–80, so the earlier "about 1.5×" claim is withdrawn.
   - A bounded inflation leg λ_π was added to the present-value identity.
   - The d_profits and s_pshare labels are fixed: FRED CP is book profits without IVA/CCAdj.
3. Each channel gets one carrier. A netting rule (rule 18) stops the learned loadings and the identities from counting the same channel twice.

Also added:
- An oil-cause label x_oil_cause (demand or supply), stated by the narrative.
- Fiscal and corporate-interest identities: F-DEBT, F-INT, V-INT.
- A tested sector-sum identity V-AGG. The direct us_equity equation stays as the benchmark.
- A timing class "预告路径" for announced paths.
- Taxes net of realized capital gains.
- New oracles O-A′ and O-A+L.
- New template G35 and scenario channels XC27 and XC28.
- Rules 17–21.
- Rewrites of about 45 of the 127 scenario decompositions.

Size:
- 23 asset-equation x_q fields and 29 factor-layer fields.
- 22 model states, 5 anchors, 12 views, 3 outputs.
- That is +3 fields net. The 5 policy-split equations and 8 identities are not fields.

Coverage of the 127 scenarios, 583 core pairs:
- Expressible: 76.3% (previous 76.0%).
- With a tested route to assets: 42.5% (previous 40%).
- Scenarios with every core channel expressible: 41. With every core channel tested: 8; C04 dropped off this list because the new XC27 channel has no tested route.

## Key findings

1. Policy fields are mostly reactions to the economy [v01]. Four train ends were used: 1994, 2004, 2014 and 2026. Coefficients use Newey-West (HAC) t-statistics.
   - The fed-funds reaction function has R² 0.40–0.54 (non-ZLB quarters, n = 86–175). The stabilizer equations have R² 0.27–0.65.
   - Conditioning on the raw field imputes a recession. A cut adds +0.16 to +0.20pp to Δunemployment per sd; a deficit increase adds +0.17 to +0.50pp.
   - The shock imputes 0 by construction. Out of sample it stays nearly uncorrelated with Δunemployment in the next decade (+0.24 / 0.00 / +0.07, against −0.56 / −0.43 / −0.22 for the raw field).
   - The lagged macro effects of the shocks have the right sign but are weak (|t| ≤ 2.0).
2. QE cannot be learned [v01]. Given the E0 core with training to 2014Q4, raw d_bs gives the wrong sign:
   - us_equity −1.7% per sd (t −1.8)
   - REITs −3.6% (t −2.7)
   - financials −3.0% (t −2.1)

   The residualized shock gives us_equity +0.5% (t 0.4) and REITs −2.7% (t −1.3), on only 26 quarters and 4 programs. There is no balance-sheet policy variance before 2008. QE's asset effect is therefore routed only through intervention paths on d_tp10, d_y2 and d_gz.
3. With CAPE as the multiple, the multiple is essentially the price [v03]:
   - Correlation with the price change is 0.99 at 1 quarter (n = 214), 0.97 at 4 quarters (n = 54) and 0.91 at 20 quarters (n = 11).
   - E10 explains only 2%–20% of price variance.

   P/Ê (CRSP quarter-end price over the 4-quarter mean of trailing EPS) separates the two legs at 20 quarters:
   - earnings 0.88 and multiple 1.10 of price variance
   - the multiple's correlation with price is 0.58

   Episode decompositions (earnings / multiple, log):
   - 1995–2000: +0.47 / +0.63
   - 2000–02: −0.55 (including write-downs) / −0.08
   - 2007Q4–2009Q1: −0.97 / +0.38 (a write-down artifact)
   - 2022Q4–2026Q2: +0.32 / +0.35
4. The earnings-cycle state s_ecyc = log(Ê/E10) predicts both legs over 20 quarters, with opposite signs (n = 15 non-overlapping windows, 1950–2020):
   - earnings: slope −1.14 (t −4.5); out-of-sample from 1995 +0.41
   - multiple: slope +0.90 (t 4.9); out-of-sample +0.35

   It barely predicts the real price: slope −0.24 (t −0.7), out-of-sample +0.03 [v03, v06]. The market looks through the earnings cycle. So the anchor terms −κ_e·s_ecyc (earnings) and +κ_m·s_ecyc (multiple) must be used together, and s_ecyc is not counted as a tested route to assets. Today s_ecyc is +0.33, about +1 sd.
5. Duration has to be computed from the total net payout yield, which includes buybacks [v03]. Comparing 1976–80 with 2021Q3–2026Q2:
   - On the total net payout yield (3.30% vs 3.19%), H30 goes from 19.5 to 19.7, a ratio of ×1.01.
   - On the dividend yield it would read ×1.46.
   - For H10 the ratios are ×1.00 versus ×1.14.

   Today H10 = 8.72 and H30 = 19.7. The earlier claim that today's valuation makes persistent rate shocks about 1.5× as potent is withdrawn.
6. The inflation leg, re-estimated on the new multiple [v05]. Annual Δlog(P/Ê) on the change in trend inflation:
   - 1974–1994: −0.147 per pp (t −1.7, n = 21). This is the calibration the 1995–2004 test may use.
   - 1974–2002: −0.175 (t −2.2, n = 29).
   - Full sample: −0.193 (t −3.1, n = 52, R² 0.09).

   The real-rate proxy is about 0 in every sample. For 2022, the real-rate leg (−0.23) plus the inflation leg (−0.10 to −0.11) gives about −0.33, against an actual P/Ê change of −0.365. The evidence rests on about two independent swings, so λ_π is a bounded scenario-layer parameter.
7. Oil-cause label [v02]:
   - Splitting d_oil by cause (demand = same sign as the Kilian index) passes the tier rule on the E0 + oil layer: +0.007 / +0.038 / +0.096; the 2015–26 90% interval is [+0.008, +0.161], n = 40/40/47.
   - Over the factor-round core F it is only marginal: +0.005 / +0.006 / +0.037, with lower bound 0.000.
   - Rejected: the goods-CPI split (E0: +0.003 / +0.050 / −0.009) and a metals-based label (−0.044 / −0.013 / +0.014).
   - The Kilian rule mislabels 1986Q1 as demand.
8. The sector-sum identity V-AGG matches the direct us_equity equation out of sample [v04]. Eight sector equations, including the new BusEq and "rest" equations, on F:
   - V-AGG: +0.242 / +0.388 / +0.502
   - direct: +0.232 / +0.408 / +0.486
   - 90% intervals of the difference: [−0.004, +0.025] / [−0.036, +0.003] / [+0.001, +0.028]

   Nasdaq tracks BusEq alone with out-of-sample R² 0.964 / 0.973 / 0.938. Adding retail and healthcare is worse in 1995–2004 (0.768).
9. Fiscal and corporate accounting:
   - The debt-accumulation identity tracks debt held by the public: correlation 0.94 for 4-quarter changes (n = 52) and 0.91 for 5-year changes (n = 11) [critic k08].
   - Interest burden: the identity (5-year mean of the 10-year yield × debt held by the public) gives 3.57% of GDP in 2026Q2. NIPA gives 3.96% and OMB gross interest gives 3.94% (FY2025, FYOIGDA188S, downloaded this round). Both published series are broader than market-debt interest [d01].
   - Falling net interest accounts for 34% of the 1989–2019 rise in the corporate profit share [critic k07].
10. Coverage of the 127 scenarios uses 583 core pairs: the earlier 576 plus XC27 (5 scenarios) and XC28 (2 scenarios).
   - Expressible: 76.3% (previous 76.0%).
   - With a tested route to assets: 42.5% (previous 40%).
   - Scenarios with every core channel expressible: 41. With every core channel tested: 8 (C04 dropped off this list because XC27 has no tested route).
   - By group, expressible / tested: A 81% / 39%, B 75% / 44%, C 90% / 58%, D 70% / 35%.
11. Size of the table:
   - Fields: 23 asset-equation x_q, 29 factor-layer x_q, 22 model states, 5 anchors, 12 views and 3 outputs. The net change is +3 fields: s_ecyc, x_oil_cause and y_eps.
   - s_pe and s_nfc_debt moved from views to identity states.
   - The 5 policy-split equations and 8 identities are not fields.
   - Rejected with reasons: a separate E0* class, realized index EPS as a quarterly input, λ_π as a learned input, anticipated paths as first-version training inputs, and Nasdaq mapped on BusEq plus retail and healthcare.

## The table (compact, as returned)

最终表（schema-revise-2；单元格截短，全文见 SCHEMA.md 第 2 节）：资产方程 x_q 23 个、因子层 29 个、模型状态 22 个、锚 5 个、视图 12 个、输出 3 个。内生性 E0 外生｜E0p 政策（系统 + 冲击，do 算子）｜E1c 商品价格｜E1 市场价格｜E2 调查｜期初｜参数。定价通道 CF 现金流、RR 实际利率、INF 通胀、RP 风险溢价、CMD 商品、SEC 行业。

| 层/档 | 字段 | 中文名 | 定义（单位） | 时点；发布滞后 | 内生性；进入 | 实测序列；覆盖 | 文献 | 资产 ← 定价通道｜路径 | 检验路径 |
|---|---|---|---|---|---|---|---|---|---|
| L0/1 | `s_cape` | 起点估值：log 席勒 CAPE | 标普综合指数实际价格 / 过去 10 年实际盈利均值，取对数（log） | 期初；E 滞后约 3 个月 | 期初；恒等式 | Shiller ie_data（shillerda…；1973Q1–2026Q3 | Campbell–Shiller 19… | 长期估值锚：给情景作者的历史分位与“回到均值”假设；按模拟的实际盈利路径用 40 季（120 个月）均…｜恒等式：V-CAPE（E10 的精确算术）… | 否 |
| L0/2 | `s_fcfy` | 自由现金流收益率（非金融公司） | 4 季均值［税后利润（含 IVA/CCAdj）+ 固定资本消耗 − 资本开支］/ 非金融公司股票市值（…（%） | 期初；NIPA 与 Z.1 … | 期初；学习 | NIPA W328RC1Q027SBEA、B456…；1973Q1–2026Q3 | 用户点名（自由现金流比）；Lettau… | 全部股票 RP/CF：估值状态｜期初状态 | 否 |
| L0/1 | `s_pd` | log 市价/股息（CRSP；各行业自… | CRSP 市值加权指数价格 / 过去 12 个月股息，取对数；行业用 French 12 行业与 Na…（log） | 期初；0（月末价格与股息） | 期初；恒等式 | Ken French CRSP 价格与股息；Fre…；1973Q1–2026Q3 | Campbell–Shiller 19… | 收入腿：总回报 = 价格回报 + D/P；各行业自身 P/D 进行业倍数；久期不再用 D/P（回购替代…｜恒等式：收入腿与行业倍数 | 否 |
| L0/1 | `s_pe` | 起点倍数：log P/Ê（CRSP 季… | CRSP 季末价格 / 标普滚动 12 个月每股盈利的 4 季均值，取对数；= s_cape − s_…（log） | 期初；Shiller E 滞… | 期初；恒等式 | valuation 组 us_equity_lm_…；1973Q1–2026Q3 | Campbell–Shiller 19… | 倍数的水平：情景作者对“市盈率回到 X”的假设落在这里；y_mult 的有界路径以它为起点｜恒等式：倍数水平（价格 = 盈利 × 倍数） | 否 |
| L0/2 | `s_payout` | 总净派现收益率（股息 + 回购 − 新… | Z.1 非金融公司（净股息 − 股票净发行）的 4 季流量均值 / 股票市值，再取 20 季均值（久期…（%） | 期初；Z.1：期初已知 q-2 | 期初；恒等式 | Z.1 BOGZ1FA106121075Q、NCB…；1973Q1 | Boudoukh–Michaely–R… | 现值久期 H(s_payout, N) 的 ρ = 1/(1 + 总派现收益率)（有回购时价格对贴现率…｜恒等式：久期 H（V-PV）与稀释（V-E… | 否 |
| L0/1 | `s_rstar` | 中性实际利率 r*（HLW 单边估计） | Holston–Laubach–Williams 美国 r* 单边估计（%） | 期初；回测必须用实时估计（当… | 期初；锚 | NY Fed HLW current estima…；1973Q1–2026Q3 | Holston–Laubach–Wil… | 美债 RR；长久期股票（纳指、REITs、公用事业）RR；估值的贴现率锚｜锚：政策利率 → r* + π_lr + … | 否 |
| L0/2 | `s_trend_g` | 趋势增长（HLW g） | HLW 模型的美国趋势增长率（年化）；备选 CBO 潜在 GDP 增速（有修订）（%） | 期初；期初已知 q-2 | 期初；锚 | NY Fed HLW current estima…；1973Q1–2026Q2 | Holston–Laubach–Wil… | 全部股票 CF（长期增长）；美债 RR（经由 r*）｜锚：现值恒等式的 g = s_trend_… | 否 |
| L0/1 | `s_infexp_lr` | 长期通胀预期（锚定程度） | 密歇根 5–10 年通胀预期中位数季均（1990Q2 起连续；1979–89 只有 41 个月）；SP…（%） | 期初；0（调查月） | 期初；锚 | UMich tbmpx1px5（PX5_MD）；P…；1990Q2–2026Q3 | Coibion–Gorodnichen… | 美债 INF；黄金 INF+；高通胀体制下股票 RP−（股债正相关）｜锚：π_lr（政策利率、10 年期与现值恒… | 否 |
| L0/1 | `s_real_y10` | 10 年实际收益率 | 2003 年起 10 年 TIPS 收益率（DFII10）；之前用 10 年名义收益率减适应性通胀预期（%） | 期初；0 | 期初；学习 | FRED DFII10（2003 起）；面板 y1…；1973Q1–2026Q3 | Campbell–Shiller–Vi… | 黄金 RR−；长久期股票 RR−；估值的贴现率｜学习：条件状态；恒等式：现值恒等式的实际利… | 否 |
| L0/2 | `s_policy_gap` | 政策立场缺口（实际政策利率 − r*） | s_ff − s_infexp1 − s_rstar（派生，不是新序列）；草表的 policy_sta…（pp） | 期初；同组成部分 | 期初；锚 | 由 s_ff、s_infexp1、s_rstar …；1978Q1（之前用 EWMA 通胀） | Taylor 1993；Laubach… | 现金、美债 RR；全部股票 RR｜锚：政策利率路径相对 r* + π 的偏离… | 否 |
| L0/2 | `s_nfc_debt` | 非金融企业债务/GDP | 非金融企业债务证券与贷款 / 名义 GDP（私募信贷没有公开数据，用它近似）（%） | 期初；Z.1：期初已知 q-2 | 期初；恒等式 | Z.1 BCNSDODNS / GDP；1973Q1–2026Q2 | Bernanke–Gertler–Gi… | 企业利息腿 V-INT：Δ(税后利润/GVA) ≈ −(1−τ)·Δ[(债务/GVA)·r_eff]，…｜恒等式：V-INT（利率 → 企业净利息 … | 否 |
| L0/1 | `s_pshare` | 税后企业账面利润/GDP 水平 | NIPA 税后企业账面利润（FRED CP，不含 IVA/CCAdj）/ 名义 GDP（%） | 期初；期初已知 q-2 | 期初；恒等式 | NIPA CP / GDP；1973Q1–2026Q3 | Greenwald–Lettau–Lu… | 股票现金流的长期水平：y_eps ≈ Δlog CP（= 名义 GDP 增长 + Δlog s_psh…｜恒等式：盈利桥 V-EARN 与利息腿 V… | 否 |
| L0/1 | `s_w` | 行业市值权重（指数构成） | French 12 行业的市值权重（公司数 × 平均市值），上季末；建模的 6 个行业（能源、工业、必…（%） | 期初；0（月末价格） | 期初；恒等式 | MAIN data/raw/french/12_I…；1926Q3–2026Q3 | —（会计恒等式） | 美股大盘 = Σ w_{i,t−1} × r_i（无拟合参数）；纳指挂在 BusEq 上｜恒等式：行业加总 | 是 |
| L0/1 | `s_oil_real` | 实际油价水平（log WTI/CPI） | 上季末 WTI / CPI，取对数（log） | 期初；CPI 滞后约 2 周 | 期初；锚 | 面板 lvl_oil、lvl_cpi；1973Q1–2026Q3 | Hamilton 2009；Baume… | 原油、能源：实际油价向长期水平回归（锚定块，a_oil 可移位）｜锚：实际油价 → 训练期中位数 + a_o… | 是 |
| L0/1 | `s_usd_real` | 实际美元水平（广义实际汇率） | 美联储实际广义美元指数（TWEXBPA，1973–2019）在 2006 年按均值比接到 RTWEXB…（log） | 期初；约 1 个月 | 期初；锚 | FRED TWEXBPA、RTWEXBGS；1973Q1–2026Q3 | Rogoff 1996；Engel 2… | 美元（因子）向长期实际水平回归；经由 d_usd 到全部资产｜锚：实际美元 → 训练期均值 + a_usd | 是 |
| S/1 | `s_core_yoy` | 核心 CPI 同比（派生：高通胀体制 … | 上月核心 CPI 同比；s_infl_regime = 1{上月核心 CPI 同比 > 4%}（FWT…（% / 0-1） | 期初；1 个月 | 期初；学习 | BLS CPILFESL；1973Q1–2026Q3 | Campbell–Sunderam–V… | 体制条件：核心 > 4% 时股债相关转正｜期初状态 | 是 |
| S/1 | `s_unrate` | 失业率 | 上月失业率（%） | 期初；1 个月 | 期初；学习 | BLS UNRATE；1973Q1–2026Q3 | Okun 1962 | 周期股、金融 CF（均值回归锚）｜学习：条件状态；锚：失业率 → a_ust… | 否 |
| S/1 | `s_ff` | 政策利率水平（联邦基金利率） | 上季末月的联邦基金利率（月均）；不再拼接影子利率；下限 0（%） | 期初；0 | 期初；学习 | FRED FEDFUNDS；1973Q1–2026Q3 | Taylor 1993 | 现金 RR；利率变化按当前水平缩放｜学习：条件状态（零下限时 = 0）；锚：→… | 否 |
| S/1 | `s_term` | 期限利差（10 年 − 政策利率） | 上季末 10 年期收益率 − 联邦基金利率（pp） | 期初；0 | 期初；学习 | FRED DGS10、FEDFUNDS；1973Q1–2026Q3 | Estrella–Mishkin 19… | 金融 CF（净息差）｜期初状态 | 否 |
| S/2 | `s_infexp1` | 短期通胀预期水平（密歇根 1 年） | 上季末密歇根 1 年通胀预期；1978 年前用适应性代理（%） | 期初；0 | 期初；学习 | FRED MICH；1978Q1 | Coibion–Gorodnichen… | 实际利率的计算腿｜经由 s_policy_gap | 否 |
| S/1 | `s_spread` | 信用利差水平（Baa − 10 年） | 上季末 Baa 与 10 年期国债收益率之差，两条腿取同一时点（1962–1985 取当月最后一个周值…（pp） | 期初；0（观测值，不修订） | 期初；学习 | FRED WBAA、WGS10YR（至 1985）…；1973Q1–2026Q4 | Gilchrist–Zakrajšek… | 全部股票、信用 RP 的起点；利差的长期锚 a_spread｜学习：条件状态；锚：→ a_spread | 否 |
| S/1 | `s_ecyc` | 盈利周期位置 log(Ê/E10) | 4 季平均的滚动 12 个月实际 EPS 相对过去 10 年实际 EPS 均值，取对数（= s_cap…（log） | 期初；Shiller E 滞… | 期初；锚 | Shiller ie_data 2026（valu…；1947Q1–2026Q2 | Campbell–Shiller 19… | 决定盈利腿与倍数腿怎么分：盈利向周期均值回归（−κ_e·s_ecyc），倍数同时反向（+κ_m·s_e…｜锚：盈利周期 → 训练期均值（两条腿相反的… | 否 |
| L1/1 | `d_ff` | 政策利率变动（联邦基金利率） | 季末月的联邦基金利率月均减上季末月；水平不低于 0；零下限期间的宽松不放在这里，而经由 d_y2、d_…（pp/季） | 同季实现/预告；0 | E0p；拆分 | FRED FEDFUNDS；1973Q1–2026Q3 | Taylor 1993；Romer–R… | 现金 RR+；美债 RR−；全部股票 RR−（纳指、REITs、公用事业久期更长）；金融 SEC±；美…｜直接：全部（F 核心因子 ff） | 是 |
| L1/2 | `d_bs` | 央行资产负债表变动（QE/QT） | 基础货币 / 名义 GDP 的季度变动；备选 Z.1 央行总资产、WALCL（2002-12 起）；冲…（pp GDP/季） | 同季实现/预告；0 | E0p；拆分 | FRED BOGMBASE / GDP；1973Q1–2026Q3 | Gagnon et al. 2011；… | 美债 RP−（期限溢价）；股票、信用 RP+；美元 RR−｜经由 d_y2, d_tp10, d_gz | 否 |
| L1/2 | `d_deficit` | 财政脉冲（联邦赤字/GDP 变动） | 联邦政府净借款（−FGDEF）/ GDP 的季度变动；拆分：系统部分 = 自动稳定器（同季与上季 Δ失…（pp GDP/季） | 同季实现/预告；约 2 个月 | E0p；拆分 | NIPA FGDEF / GDP；1973Q1–2026Q2 | Blanchard–Perotti 2… | 周期股 CF+；美债 RP（供给）；美元 RR+（Mundell–Fleming）｜经由 d_rgdp, d_cons_*, … | 否 |
| L1/2 | `d_transfers` | 转移支付变动 | 个人经常性转移收入 / 个人收入 的季度变动；拆分：系统部分 = 自动稳定器（同季与上季 Δ失业率、Δ…（pp/季） | 同季实现/预告；1 个月 | E0p；拆分 | FRED PCTR / PI；1973Q1–2026Q3 | Oh–Reis 2012；Parker… | 必需消费、REITs CF+；商品通胀 INF+｜经由 d_cons_goods, d_cp… | 否 |
| L1/2 | `d_defense` | 国防支出变动 | 联邦国防消费与投资 / GDP 的季度变动（pp GDP/季） | 同季实现/预告；1 个月 | E0；学习 | FRED FDEFX / GDP；1973Q1–2026Q3 | Ramey 2011；Ramey–Zu… | 工业（军工）CF+；美债 RP｜经由 d_rgdp, d_deficit | 否 |
| L1/1s | `d_corptax` | 企业有效税率变动 | 100 ×（1 − 税后利润 / 税前账面利润）的季度变动（NIPA）（pp/季） | 同季实现/预告；约 2 个月 | E0；学习 | NIPA CP、A053RC1Q027SBEA；1973Q1–2026Q2 | McGrattan–Prescott … | 全部股票 CF−（税后现金流；长期水平）；REITs SEC（穿透征税，税率上升时相对受益）｜直接：医疗、能源、美股（弱）；长期经由 s… | 是 |
| L1/2 | `d_perstax` | 个人所得税负担变动 | 个人经常性税收 / 个人收入 的季度变动；拆分：系统部分 = 过去 4 季股票回报（已实现资本利得的税…（pp/季） | 同季实现/预告；1 个月 | E0p；拆分 | NIPA W055RC1Q027SBEA / PI；1973Q1–2026Q3 | Romer–Romer 2010；Me… | 必需消费、REITs CF−；美债 RP+｜经由 d_cons_*, d_deficit | 否 |
| L1/2 | `d_tariff` | 有效关税率变动 | 海关关税收入 / 商品进口额 的季度变动（pp/季） | 同季实现/预告；1–3 个月 | E0；学习 | NIPA B235RC1Q027SBEA / A2…；1973Q1–2026Q3 | Amiti–Redding–Weins… | 必需消费、工业 SEC−（投入成本）；商品通胀 INF+；美元 ±（叙事须写明）｜经由 d_cpi_goods, d_epu… | 否 |
| L1/1s | `d_sent` | 消费者信心变动（密歇根） | 密歇根消费者信心季末值的变动；1978 年前季度调查保持（指数点/季） | 同季预期；0（月中初值） | E2；学习 | FRED UMCSENT；1973Q1–2026Q3 | Carroll–Fuhrer–Wilc… | 金融 CF/RP+；全部股票 RP+；必需消费（相对）−｜直接：金融 | 是 |
| L1/2 | `d_hpi` | 房价增长 | Freddie Mac 全国房价指数（季调）的年化增长；1975 年前用 Shiller 拼接（%年化） | 同季实现；约 2 个月 | E1；学习 | Freddie Mac FMHPI US SA；1975Q2–2026Q2 | Leamer 2007；Mian–Su… | 金融、REITs CF+；消费（财富效应）；服务通胀（住房，滞后 12–18 月）｜经由 d_cons_goods, d_cp… | 否 |
| L1/2 | `x_lending_std` | 银行放贷标准（信贷供给） | SLOOS：对大中型企业收紧工商业贷款标准的银行净占比（季度调查；它本身已是“变化”，所以用水平；文中…（%） | 滞后一季；FRED 把 4 月调… | E2；学习 | FRED DRTSCILM（Fed SLOOS）；1990Q2–2026Q3 | Lown–Morgan 2006；Ba… | 金融、REITs CF−；全部股票 RP−｜直接：REITs、金融、美股 | 否 |
| L1/1 | `d_igrea` | 全球实际活动（外需） | Kilian 全球实际经济活动指数季均的变动（指数/季） | 同季实现；约 1 个月 | E1c；学习 | FRED IGREA；1973Q1–2026Q2 | Kilian 2009；Baumeis… | 工业、金融 CF+；能源 CMD（需求推动的油价）｜直接：工业、金融、美股、纳指、REITs | 是 |
| L1/1s | `d_usd_em` | 美元对新兴市场货币（EM 压力） | 美元对其他重要贸易伙伴指数（月均，至 2005-12）接美元对新兴经济体指数（月末，2006-01 起…（log%/季） | 同季预期；0 | E1；学习 | FRED TWEXOMTH、DTWEXEMEGS；1973Q2–2026Q3 | Bruno–Shin 2015；Avd… | 工业、能源 CF−（外需）；金融（敞口）｜直接：工业、美股、能源 | 是 |
| L1/2 | `d_bund` | 全球利率（德国 10 年期变动） | 德国 10 年期国债收益率的季度变动；OECD 月度数据是月均，所以是季末月月均对上季末月月均（不是月…（pp/季） | 同季预期；0；季均口径（月均） | E1；学习 | FRED IRLTLT01DEM156N；1973Q1–2026Q3 | Jotikasthira–Le–Lun… | 美债 RR/RP；美元｜经由 d_y10, d_tp10, d_u… | 否 |
| L1/1s | `d_metals_oil` | 基础金属/油价（区分油价的供给与需求） | 世界银行基础金属指数 / 原油均价 的季度对数变动（log%/季） | 同季实现；约 1 个月 | E1c；学习 | World Bank Pink Sheet（因子轮…；1973Q1–2026Q3 | Kilian 2009；Baumeis… | 能源 CMD（需求推动的油价上涨对能源股更有利）；原油（区分供需）｜直接：能源（FWT 核心） | 是 |
| L1/1 | `x_oil_cause` | 油价变动的原因（需求 / 供给） | 本季油价变动的原因标签：需求 = 油价与 Kilian 全球活动同向，供给 = 反向（训练时按这条无参…（0/1） | 同季实现；0（叙事）；训练时同 … | E0；学习 | FRED IGREA 与面板 f_oil 的符号（…；1973Q2–2026Q2 | Kilian 2009；Kilian–… | 全部股票 CMD/CF：需求推动的油价上涨与股价同向、供给推动的反向；能源股两者都正｜直接：全部股票（与 d_oil 的交互） | 是 |
| L1/1s | `d_natgas` | 天然气价格 | 天然气 PPI 季均的对数变动；亨利港现货（EIA，1997 起）作备选（log%/季） | 同季实现；1 个月 | E1c；学习 | FRED WPU0531；1973Q1–2026Q3 | Hartley–Medlock–Ros… | 公用事业 CF±（燃料成本与费率传导）；能源 CMD+｜直接：公用事业、能源 | 是 |
| L1/1 | `x_supply_chain` | 供应链压力 | 费城联储制造业调查交货时间扩散指数季均（1968-05 起；扩散指数本身已是“变化”，用水平，文中 s…（扩散指数） | 同季实现；0（调查月） | E0；学习 | Philadelphia Fed MBOS DTC…；1973Q1–2026Q3 | Benigno et al. 2022… | 工业 SEC−（缺件）；商品通胀 INF+｜直接：工业、金融、REITs、必需消费（T… | 是 |
| L1/2 | `d_itinv` | IT 设备与软件实际投资增长（通用技术… | 实际信息处理设备与软件投资（数量指数）的年化增长（%年化） | 同季实现；约 1 个月 | E0；学习 | FRED B679RA3Q086SBEA；1973Q1–2026Q3 | Jorgenson–Stiroh 20… | 纳指 CF+；公用事业（电力需求）CF+；工业 CF+｜直接：纳指 | 否 |
| L1/1s | `d_lprod` | 劳动生产率增长（非农） | 非农企业部门每小时产出的年化增长（%年化） | 同季实现；约 5 周 | E0；学习 | BLS OPHNFB（MAIN 已有）；1973Q1–2026Q2 | Fernald 2014；Basu–F… | 纳指、全部股票 CF+（长期增长）；美债 RR（经由 r*）｜直接：纳指、美股、医疗（T2/T3） | 是 |
| L1/2 | `d_lf` | 劳动力增长（含人口与移民） | 民用劳动力季均的年化增长；每年 1 月人口控制数造成断点，用 4 季平均或剔除 1 月跳变；参与率变动…（%年化） | 同季实现；约 1 周 | E0；学习 | BLS CLF16OV（CNP16OV 作人口腿）；1973Q1–2026Q3 | Galí–Smets–Wouters … | 经由工资与服务通胀；长期经由趋势增长｜经由 d_wage, d_cpi_serv… | 否 |
| L1/2 | `d_lshare` | 劳动收入占比变动 | 非农企业部门劳动收入占比（指数）的季度对数变动；与单位劳动成本是恒等式，二者只留一个（log%/季） | 同季实现；约 5 周（FWT 按… | E0；学习 | BLS PRS85006173；1973Q1–2026Q2 | Karabarbounis–Neima… | 全部股票 CF−（利润率，主要是长期水平）；服务通胀 INF+｜经由 d_cpi_serv, s_lsha… | 否 |
| L1/2 | `d_cpbill` | 融资市场压力（商业票据 − 国库券利差… | 3 个月商业票据（1996-12 前 CP3M，之后金融 CP CPF3M）减 3 个月国库券，季均的…（pp/季） | 同季预期；0 | E1；学习 | FRED CP3M、CPF3M、TB3MS；1973Q1–2026Q3 | Friedman–Kuttner 19… | 金融 CF−；REITs RP−；信用 RP+｜经由 d_gz | 否 |
| L1/1 | `d_bdlev` | 券商（非银）杠杆变动 | Z.1 券商总资产 /（资产 − 负债）的季度对数变动（log%/季） | 同季实现；Z.1 约 10 周 | E1；学习 | Z.1 L.130（BOGZ1FL66409000…；1973Q1–2026Q2 | Adrian–Shin 2010；Ad… | 金融 CF；全部风险资产 RP（强制去杠杆）｜直接 | 是 |
| L1/1 | `d_epu` | 经济政策不确定性变动 | Baker–Bloom–Davis 新闻 EPU 季均的对数变动（1985 年前历史指数）（log%/季） | 同季预期；1 个月 | E2；学习 | PolicyUncertainty.com；1973Q1–2026Q3 | Baker–Bloom–Davis 2… | 全部股票 RP−；黄金 RP+；工业 CF−（投资推迟）｜直接：全部股票（T1、T2） | 是 |
| L1/2 | `d_gpr` | 地缘政治风险变动 | Caldara–Iacoviello 历史 GPR 指数季均的对数变动（log%/季） | 同季预期；1 个月 | E2；学习 | matteoiacoviello.com GPRH；1973Q1–2026Q3 | Caldara–Iacoviello … | 原油、能源 CMD+（预防性需求）；黄金 RP+；工业（军工）CF+；全部股票 RP−｜直接：能源、工业 | 否 |
| L1/2 | `d_epu_health` | 医疗政策不确定性变动（医疗行业） | 分类 EPU“医疗”子指数季均的对数变动（log%/季） | 同季预期；1 个月 | E2；学习 | PolicyUncertainty.com Cat…；1985Q2–2026Q3 | Baker–Bloom–Davis 2… | 医疗 RP−（药价与医疗监管风险）｜直接：医疗 | 否 |
| L1/2 | `d_infexp1` | 短期通胀预期变动（密歇根 1 年） | 密歇根 1 年通胀预期季均的变动；1978 年前用面板适应性代理（pp/季） | 同季预期；0（调查月） | E2；学习 | FRED MICH；1978Q2–2026Q3 | Coibion–Gorodnichen… | 美债 INF−；黄金 INF+｜经由 d_wage, d_y2, d_y10 | 否 |
| L1/2 | `d_infexp_lr` | 长期通胀预期变动（去锚/再锚的新闻） | 密歇根 5–10 年通胀预期中位数季均的变动（1990Q2 起）（pp/季） | 同季预期；0 | E2；学习 | UMich tbmpx1px5 PX5_MD；SP…；1990Q3–2026Q3 | Coibion–Gorodnichen… | 美债 INF−；黄金 INF+；公用事业 RR｜直接：黄金、公用事业 | 否 |
| L2/2 | `d_rgdp` | 实际 GDP 增长 | 实际 GDP 的年化增长（%年化） | 同季实现；约 1 个月（初值） | E0；学习 | BEA GDPC1；1973Q1–2026Q3 | — | 周期股 CF+｜经由 d_ip（F 的核心因子） | 否 |
| L2/1 | `d_ip` | 工业产出增长 | 工业产出季末对数变动（log%/季） | 同季实现；约 2 周 | E0；学习 | Fed G.17 INDPRO；1973Q1–2026Q3 | — | 工业、能源 CF+｜直接：全部（F 核心因子） | 是 |
| L2/1 | `d_unrate` | 失业率变动 | 失业率季末变动（pp/季） | 同季实现；约 1 周 | E0；学习 | BLS UNRATE；1973Q1–2026Q3 | Okun 1962 | 周期股、金融 CF−｜直接：全部（F 核心因子） | 是 |
| L2/1 | `d_cpi_goods` | 商品通胀 | CPI 商品季度对数变动（年化）（%年化） | 同季实现；约 2 周 | E0；学习 | BLS CUSR0000SAC；1973Q1–2026Q3 | — | 名义资产 INF−；能源 CMD+｜直接：全部（F 核心因子） | 是 |
| L2/1 | `d_cpi_serv` | 服务通胀 | CPI 服务季度对数变动（年化）（%年化） | 同季实现；约 2 周 | E0；学习 | BLS CUSR0000SAS；1973Q1–2026Q3 | — | 美债 INF−；长久期股票 RR−｜直接：全部（F 核心因子） | 是 |
| L2/2 | `d_core` | 核心通胀 | 核心 CPI 季度对数变动（年化）；用来滚动更新 s_core_yoy 与体制标志（%年化） | 同季实现；约 2 周 | E0；学习 | BLS CPILFESL；1973Q1–2026Q3 | — | 经由体制状态｜经由 s_infl_regime | 否 |
| L2/2 | `d_wage` | 工资增长 | 生产与非管理人员平均时薪季末年化增长（%年化） | 同季实现；约 1 周 | E0；学习 | BLS AHETPI；1973Q1–2026Q3 | Blanchard–Galí 2007 | 劳动密集行业 CF−；服务通胀 INF+｜经由 d_cpi_serv | 否 |
| L2/1 | `d_oil` | 油价变动 | WTI 季末对数变动（1986 年前为月均）（log%/季） | 同季实现；0 | E1c；学习 | 面板 f_oil（DCOILWTICO / WTI…；1973Q1–2026Q3 | Hamilton 1983；Kilia… | 能源、原油 CMD+；工业、必需消费 SEC−；商品通胀 INF+｜直接：全部（F 核心因子） | 是 |
| L2/2 | `d_y2` | 2 年期收益率变动（预期政策路径） | 2 年期国债季末收益率变动；预期政策路径的变化 = d_y2 − d_ff（两条腿都留，不另设字段）（pp/季） | 同季预期；0 | E1；学习 | FRED DGS2（月末）；1976Q3–2026Q3 | Gürkaynak–Sack–Swan… | 全部股票 RR−；美元 RR+；美债 RR−｜经由 d_y10, d_usd | 否 |
| L2/1 | `d_y10` | 10 年期收益率变动 | 10 年期国债季末收益率变动（pp/季） | 同季预期；0 | E1；学习 | FRED DGS10（月末）；1973Q1–2026Q3 | Ang–Piazzesi 2003 | 美债 RR−；长久期股票、公用事业、REITs RR−｜直接：全部（F 核心因子） | 是 |
| L2/1 | `d_tp10` | 10 年期限溢价变动（ACM） | ACM 10 年期限溢价季末变动；核心通胀 > 4% 时的交互项已在 F 中（pp/季） | 同季预期；0 | E1；学习 | NY Fed ACMTP10；1973Q1–2026Q3 | Adrian–Crump–Moench… | 美债 RP−；高通胀体制下股票 RP−；公用事业 RR−｜直接：公用事业；体制交互进入全部（FWT） | 是 |
| L2/1 | `d_gz` | GZ 信用利差变动 | GZ 利差（含违约风险）的季度变动（pp/季） | 同季预期；0 | E1；学习 | FRB ebp_csv gz_spread；1973Q2–2026Q3 | Gilchrist–Zakrajšek… | 全部股票 RP−；金融、工业 CF−；风险偏好的信用部分（恐慌）从这里进入｜直接：全部（FWT 新核心因子） | 是 |
| L2/1 | `d_spread` | Baa − 10 年利差变动（计时修正… | Baa 与 10 年期同一时点取值的利差变动（FWT D1 修正）（pp/季） | 同季预期；0 | E1；学习 | FRED DBAA/WBAA、DGS10/WGS1…；1973Q1–2026Q3 | — | 全部股票 RP−｜直接：全部（F 核心因子） | 是 |
| L2/1 | `d_usd` | 美元变动 | 贸易加权美元季末对数变动（log%/季） | 同季预期；0 | E1；学习 | 面板 f_usd（Fed 广义/主要货币）；1973Q2–2026Q3 | Lustig–Roussanov–Ve… | 纳指、必需消费 CF−（海外收入）；黄金、原油 CMD−；工业 CF−｜直接：全部（F 核心因子） | 是 |
| L2/1s | `d_tips10` | 10 年 TIPS 实际收益率变动（黄… | 10 年 TIPS 收益率季末变动；名义 = 实际 + 盈亏平衡通胀（pp/季） | 同季预期；0 | E1；学习 | FRED DFII10；2003Q1 | Erb–Harvey 2013 | 黄金 RR−；公用事业 RR−；股票倍数的实际利率腿（现值恒等式，H × Δr）｜直接：黄金、公用事业（2003 年起）；恒… | 是 |
| L2/2 | `d_cons_goods` | 实际商品消费增长 | 实际商品消费（数量指数）季均的年化增长（%年化） | 同季实现；约 1 个月 | E0；学习 | BEA DGDSRA3M086SBEA；1973Q1–2026Q3 | Guerrieri–Lorenzoni… | 必需消费、工业 CF+；商品通胀 INF+｜经由 d_cpi_goods, d_ip | 否 |
| L2/2 | `d_cons_serv` | 实际服务消费增长 | 实际服务消费（数量指数）季均的年化增长（%年化） | 同季实现；约 1 个月 | E0；学习 | BEA DSERRA3M086SBEA；1973Q1–2026Q3 | Guerrieri et al. 20… | REITs、能源 CF+；服务通胀 INF+｜经由 d_cpi_serv, d_unra… | 否 |
| L2/2 | `d_capex` | 实际企业投资增长 | 实际非住宅固定投资（数量指数）的年化增长（%年化） | 同季实现；约 1 个月 | E0；学习 | BEA A008RA3Q086SBEA；1973Q1–2026Q3 | Justiniano–Primicer… | 工业、纳指 CF+｜经由 d_ip, d_rgdp | 否 |
| L2/2 | `d_resinv` | 实际住宅投资增长 | 实际住宅投资（数量指数）的年化增长（%年化） | 同季实现；约 1 个月 | E0；学习 | BEA A011RA3Q086SBEA；1973Q1–2026Q3 | Leamer 2007 | 工业、金融 CF+｜经由 d_rgdp | 否 |
| L2/2 | `d_exports` | 实际出口增长 | 实际出口（链式美元）的年化增长（%年化） | 同季实现；约 1 个月 | E0；学习 | BEA EXPGSC1；1973Q1–2026Q3 | — | 工业 CF+｜经由 d_rgdp | 否 |
| L2/2 | `d_profits` | 企业税后账面利润增长（NIPA CP） | NIPA 税后企业利润（FRED CP：不含 IVA 与 CCAdj 的账面利润，含存货持有收益；比含…（log%/季） | 同季实现；约 2 个月（第二次 … | E0；学习 | BEA CP（FRED CP；CPATAX 作经济…；1973Q1–2026Q2 | Greenwald–Lettau–Lu… | 全部股票 CF+：经由盈利桥 V-EARN 到指数盈利 y_eps｜经由 y_eps（V-EARN）, s_p… | 否 |
| L2/2 | `d_cre` | 商业地产价格变动 | Z.1 商业地产价格指数的季度对数变动（评估平滑）（log%/季） | 同季实现；Z.1 约 10 周 | E1；学习 | Z.1 BOGZ1FL075035503Q；1973Q1–2026Q2 | Geltner 1993（评估平滑） | REITs CF；金融 CF（信用损失）｜经由 d_chargeoff | 否 |
| L2/2 | `d_chargeoff` | 银行贷款核销率变动（信用损失） | 全部商业银行全部贷款核销率（年化）的季度变动（pp/季） | 同季实现；约 2 个月 | E0；学习 | FRED CORALACBS；1985Q2–2026Q2 | Bernanke–Lown 1991 | 金融 CF−；REITs CF−｜直接：金融 | 否 |
| 输出/— | `y_assets` | 13 个资产的季度对数回报（联合分布） | 季末相对季初的对数价格回报 + 收入；模型的输出（log） | 同季实现；0 | 输出；输出 | 面板 p_*；1973Q1 | — | —｜— | — |
| 输出/— | `y_mult` | 倍数变化 Δlog(P/Ê)（美股；各… | CRSP 季末价格对 4 季平均滚动 EPS 的对数变化：y_assets(美股) = y_eps +…（log） | 同季实现；0 | 输出；输出 | valuation 组 us_equity_lm_…；1973Q1 | Campbell–Shiller 19… | —｜— | — |
| 输出/— | `y_eps` | 盈利腿 Δlog Ê（指数每股盈利，4… | 标普滚动 12 个月 EPS 的 4 季均值的对数变化；价格 = 盈利 × 倍数：Δlog P = y…（log） | 同季实现；Shiller E 滞… | 输出；输出 | Shiller ie_data 2026 sp_le；1947Q1–2026Q2 | Campbell–Shiller 19… | —｜— | — |
| 锚/1 | `a_tp` | 期限溢价锚 | 10 年期的长期水平 = r* + π_lr + a_tp 中的 a_tp（pp） | 情景路径；— | 参数；锚 | NY Fed ACMTP10（只用于默认值）；1973 | Adrian–Crump–Moench… | 美债、长久期股票：10 年期的长期水平｜锚：锚定块（第 2.6 节） | 否 |
| 锚/1 | `a_spread` | 信用利差锚 | Baa − 10 年的长期水平（pp） | 情景路径；— | 参数；锚 | s_spread；1973 | — | 信用、全部股票：利差的长期水平｜锚：锚定块（第 2.6 节） | 否 |
| 锚/1 | `a_ustar` | 失业率锚 | 失业率的长期水平（%） | 情景路径；— | 参数；锚 | s_unrate；1973 | — | 周期股、金融：失业率的长期水平｜锚：锚定块（第 2.6 节） | 否 |
| 锚/1 | `a_oil` | 实际油价锚移位 | 实际油价长期目标（训练期中位数）的 log 移位（log） | 情景路径；— | 参数；锚 | s_oil_real；1973 | — | 原油、能源｜锚：锚定块（第 2.6 节） | 是 |
| 锚/1 | `a_usd` | 实际美元锚移位 | 实际美元长期目标（训练期均值）的 log 移位（log） | 情景路径；— | 参数；锚 | s_usd_real；1973 | — | 美元（因子），经由 d_usd 到全部资产｜锚：锚定块（第 2.6 节） | 是 |

视图（只给翻译器与界面，不进模型）：`s_debt` 联邦公众持有债务/GDP（路径由 F-DEBT 算）；`s_int_burden` 联邦公众持有债务的利息负担（恒等式口径 3.57%，2026Q2）；`s_hh_debt` 家庭债务/可支配收入；`s_foreign_hold` 外国持有美债占比；`s_lshare` 劳动收入占比水平；`s_it_share` IT 设备与软件投资/GDP；`s_old65` 65 岁以上人口占比；`s_gold_real` 实际金价水平；`s_house_pr` 房价/租金；`s_bank_dur` 银行久期错配损失；`s_tdsp` 家庭偿债负担；`wc_gap` 白领与蓝领工资增速差。

政策拆分（E0p，第 3.0 节；系统部分只用同季 E0 宏观与期初状态，拟合到 train_end）：`d_ff` → u_ff（R² 0.54/0.51/0.46/0.40，非零下限季度；零下限 u_ff = 0）；`d_bs` → u_bs（2008Q3 起；之前 ≡ 0；R² 0.56/0.68）；`d_deficit` → u_def（稳定器 R² 0.34/0.31/0.33/0.65）；`d_transfers` → u_tr（R² 0.54/0.50/0.27/0.41）；`d_perstax` → u_ptax（含过去 4 季股票回报，R² 0.18–0.22）。
恒等式：
- V-PRICE：Δlog P = y_eps + y_mult；总回报另加 D/P。P 是 CRSP 季末价格，Ê 是滚动 12 个月 EPS 的 4 季均值。
- V-EARN：y_eps ≈ Δlog CP（NIPA 税后账面利润）− 稀释 + 海外项 + 行业构成项 + 减记项 − κ_e·s_ecyc。
- V-INT：Δ(税后利润/GVA) ≈ −(1−τ)·Δ[(债务/GVA)·r_eff]，r_eff = a + b·过去 28 季 10 年期均值。
- V-PV：Δy_mult ≈ −H(s_payout, N)·(Δr_real + ΔERP) − λ_π·Δπ_trend·1{期初核心 > 4%} + H_g·Δg_e + κ_m·s_ecyc + ε。
- V-CAPE：E10 = 过去 40 季实际 Ê 的均值。
- V-AGG：us_equity = log Σ w_{i,t−1}·exp(r_i)，覆盖 8 个行业；纳指 = β·BusEq。
- F-DEBT：D/Y_{t+1} = D/Y_t·(1 − g) + 赤字/Y。
- F-INT：利息/Y = r_eff·D/Y，r_eff = 过去 20 季 10 年期均值。

情景专用：X01 股票腿（d_gz、d_epu 已隐含部分之外）至 X26 同上一版，另加 X27 指数盈利桥的情景项（海外利润、行业构成、会计减记）。
覆盖（127 个情景、583 个核心渠道）：
- 可表达 76%（上一版 76%），有检验路径 42%（上一版 40%）。
- A/B/C/D 组可表达 81%/75%/90%/70%，有检验路径 39%/44%/58%/35%。
- 全部可表达的情景 41 个，全部有检验路径 8 个。
- 可表达低于 50% 的情景：B22、B28、B62、B65、B68、B69、D23、B51、B02、B46、D26、D09、D10、D17、D30、B12、D22、B64。
- 每个情景的明细在 coverage_final.csv。

## Decomposition templates (as returned)

通用分解模板（幅度 = 季度 sd，回测时只用 train_end 以前；全文与足迹见 SCHEMA.md 第 6 节；127 个情景各自的规则在 SCHEMA.md 第 7 节与 schema.json 的 scenario_rules）：
- G01 疫情封锁与重启：
  - 封锁（1–2 季）：d_cons_serv −6 sd 以上，d_lf −4 sd，d_unrate +6 sd，d_transfers +8 sd，d_deficit +8 sd（冲击口径），d_bs +5 sd，d_ff 降到 0，d_oil −2 sd（x_oil_cause = 需求），d_gz +3~+5 sd 后回落。
  - 重启（2–4 季）：d_cons_serv +3 sd，d_cons_goods +2 sd，x_supply_chain +3~+5 sd，d_cpi_goods +1.5 sd。
  - 通胀与加息（4–8 季）。
- G02 能源生产国卷入的战争：d_gpr +2~+4 sd，d_oil +2~+4 sd，d_metals_oil −2 sd，d_epu +1 sd，d_sent −1~−2 sd，d_gz 0~+1 sd，d_cpi_goods +1.5 sd；x_oil_cause = 供给；随后 2 季油价回落，实际油价锚 s_oil_real 把冲击拉回。
- G03 石油供给冲击/禁运：d_oil +3~+4 sd，d_metals_oil −4 sd，x_supply_chain +2 sd，d_cpi_goods +2 sd，d_igrea 不升；x_oil_cause = 供给。
- G04 银行业恐慌：
  - d_gz +2~+7 sd，d_cpbill +1~+3 sd，d_bdlev −2~−5 sd，d_chargeoff +，x_lending_std +2~+3.5 sd（滞后一季），d_ff −，d_bs +，d_usd +1.5 sd。
  - 油价下跌标 x_oil_cause = 需求；商品通缩是需求崩溃，不是 1986 年式的供给性去通胀。
- G05 关税战：d_tariff +2~+10 sd/季（实施季），d_epu +1~+2 sd，d_cpi_goods +0.5~+1 sd，d_usd 由叙事写明；已公布的关税日程写预告路径。
- G06 AI/资本开支繁荣与破裂：d_itinv +1 sd/季，d_capex +，d_lprod +（滞后）；信用利差不必收窄。繁荣分两种：
  - 盈利驱动（y_eps 路径，2023–26：盈利 +0.32、倍数 +0.35）。
  - 倍数驱动（1995–2000：盈利 +0.47、倍数 +0.63，走 X01）。
  - 破裂时的减记进 V-EARN 的减记项。
- G07 财政扩张：相机财政写冲击 u_def +1~+3 sd（支票 u_tr，军备 d_defense，减税 u_ptax）；稳定器部分由模型按 d_unrate 算出；原始字段会把赤字补全成衰退。
- G08 QE/YCC/QT：u_bs（2008 起）。资产效应只经由干预路径：d_y2 −、d_tp10 −1 sd、d_gz −1~−3 sd、d_usd −。YCC 把 d_tp10 压在目标上。d_bs 本身对资产学不出。
- G09 沃尔克式去通胀：d_ff +2~+6 sd，d_y2 +（领先），d_y10 +，d_usd +1 sd/季，d_unrate 滞后 +，d_infexp1 缓慢 −。
  - 倍数的 λ_π 通胀腿在高通胀体制激活，与 TP×core4 只用一个。
  - 加息若已由写明的通胀给出（反应函数），u_ff 只是比规则更紧的部分。
- G10 房地产泡沫破裂：d_hpi −1~−3 sd（8–12 季），d_resinv −2 sd，按 G04 写银行压力，d_unrate +，d_ff −。
- G11 强美元（原因须写明）：d_usd +1~+2 sd/季；d_exports −，d_cpi_goods −，d_oil −；实际美元锚把美元拉回；盈利桥的海外项。
- G12 企业减税：d_corptax 4 季累计 −1~−4pp，s_pshare↑。长期走 V-EARN；季度 β 与 V-EARN 不同时用。
- G13 增税/整顿：u_ptax +1~+2 sd（去掉资本利得部分），d_corptax +，赤字由模型推断；泡沫后个人税的回落是资本利得，不是减税。
- G14 劳动转向资本：d_lshare −，s_pshare↑，wc_gap ±，d_lprod +；利润份额上升经由 V-EARN 到 y_eps。
- G15 新兴市场危机：d_usd_em +1~+2.5 sd，d_gz +，d_igrea −，d_y10 −，d_cpbill +。
- G16 非银去杠杆/融资螺旋：d_bdlev −1~−5 sd，d_cpbill +1~+3 sd，d_gz +2~+7 sd。
- G17 原油供给过剩/价格战：d_oil −3~−6 sd，d_metals_oil +2.5~+5 sd，d_gz +（能源信用）；x_oil_cause = 供给（2020Q1 是需求与供给叠加，叙事写明）。
- G18 期限溢价冲击：d_tp10 +0.6~+2.2 sd，d_y10 +，d_bund +；首次加息型另有 d_y2 +。
- G19 需求推动的商品繁荣：d_igrea +0.7~+1.2 sd，d_oil +，d_exports +，d_usd −；x_oil_cause = 需求。
- G20 财政整顿/和平红利：u_def −，d_defense −，u_ptax +，d_tp10 −，d_y10 −。
- G21 政策/政治不确定性：d_epu +1~+3 sd，d_gz +（伴随恐慌时），d_sent −，d_tp10 ±。
- G22 通胀预期去锚：d_infexp1 +，d_infexp_lr +，d_cpi_serv +，d_wage +，d_tp10 +；c_q 给 s_infexp_lr 上升路径；λ_π 通胀腿。
- G23 信贷紧缩：x_lending_std +1.6~+3.5 sd（描述上一季），d_chargeoff +，d_gz +，d_cre −，d_capex −。
- G24 医疗政策：d_epu_health +；立法本身作医疗行业覆盖项。
- G25 生产率浪潮：d_lprod +1~+3 sd（滞后于投资），d_itinv +，d_wage +；倍数的增长腿走 X01；c_q 给 s_trend_g、s_rstar 路径。
- G26 劳动供给/移民：d_lf ±（先剔除 1 月断点），d_wage +，d_cpi_serv +；c_q 给 s_trend_g 路径。
- G27 自然灾害与基础设施中断：d_natgas +，d_oil +，x_supply_chain +，d_cpi_goods +，d_deficit +。
- G28 行业监管：医疗 d_epu_health、银行 x_lending_std、税负 d_corptax 进字段，其余作行业覆盖项。
- G29 主权与制度风险：d_epu +2~+3 sd，d_tp10 +，d_usd −，d_infexp_lr +；c_q 移动 a_tp、a_usd、s_policy_gap；债务与利息路径由 F-DEBT、F-INT 算出。
- G30 估值重估：
  - 倍数 y_mult（P/Ê）按 V-PV 拆成四条腿：实际利率腿、ERP 腿（d_gz + X01）、λ_π 通胀腿、增长腿。
  - 盈利腿 y_eps 单独给。
  - 压力测试的“股价 −50%”拆成两条腿作有界条件（熵池），检验时掩码。
- G31 全球利率冲击：d_bund ±（月均口径），d_usd ±，d_tp10 ±，d_gz +，d_cpbill +。
- G32 日本化：d_ff 降到 0，d_bs +，通胀 −，d_unrate +，d_hpi −；c_q 给 s_rstar↓、s_trend_g↓、s_infexp_lr↓；利率下降经由 V-INT 抬高利润份额。
- G33 银行久期错配与存款挤兑：期初 s_bank_dur 高；x_lending_std +，d_bs +（BTFP），d_cre −；d_gz 可能不升；金融股与 REITs 的幅度由情景给。
- G34 加息而不衰退：d_ff +2~+3 sd/季，d_y2 +，d_y10 +；叙事写明 d_unrate ≈ 0；若加息由反应函数给出，u_ff ≈ 0。
- G35 政策干预（本轮新增）：只写干预——u_ff −（政治压力下降息）、u_def/u_tr +（刺激、支票）、u_bs +（YCC/QE，另写 d_tp10 的目标路径）。
  - 同季的产出、失业、通胀不随干预补全。
  - 之后各季按学到的滞后反应（弱：t ≤ 2.0）。
  - 可信度损失另给 c_q 路径。

本轮新增或改写的全局规则（全部 21 条见 SCHEMA.md 第 9 节）：
4. 资产价格、估值倍数、VIX、实现的指数 EPS 与用股价构造的量永远不是输入。y_mult（P/Ê）与 y_eps 是输出，检验时掩码。
16. 油价写明原因（x_oil_cause：供给或需求）；商品通胀不另拆。
17. do 算子：
   - 政策干预写冲击 u；系统方程里的同季宏观不以冲击为条件，消费与市场价格可以。
   - “因为衰退而降息”写衰退的宏观，政策由反应函数给出。
   - QE 只经由 d_tp10、d_y2、d_gz 的干预路径。
18. 一个渠道一个载体：
   - 季度上利率与 ERP 由学习载荷承担。
   - 多年上 V-PV 只作倍数累计路径的熵池目标，扣除只用字段跑出的部分。
   - X01 是 d_gz、d_epu 已隐含部分之外的剩余。
   - 税走 V-EARN 或季度 β 之一；λ_π 与 TP×core4 之一。
19. 盈利与倍数分开写：
   - 指数盈利的陈述写进 y_eps 的桥项 X27；估值的陈述写进 X01。
   - “估值不变”必须写明口径（P/Ê 或 CAPE）。
   - 只写盈利回落时，倍数按 κ_m·s_ecyc 补全。
20. 预告路径：已宣布的政策与宏观路径写进 c_q；N1 型叙事必须写出预期的政策路径。
21. 会计闭合：债务、利息、利润份额的利率部分、CAPE 都由恒等式算出；翻译器不能给出与它们矛盾的数字。

## Open questions

1. The do-operator is supported only by regressions so far. It still has to be built into the masked joint model and tested there: same-quarter output, unemployment and inflation should not be imputed from a policy shock; consumption and market prices may be; macro responds to the shock with a lag. The masked model's imputation-accuracy test was not run either.
2. The reaction-function and stabilizer equations use current-vintage data. The ALFRED real-time re-estimate was not done, and setting the fed-funds shock to 0 in zero-lower-bound quarters is a simplification.
3. QE has only 26 quarters and 4 programs (2008Q3–2014Q4). The intervention magnitudes in template G08 can only come from the 2008–14 footprints.
4. The inflation leg λ_π rests on about two independent swings. A 1995–2004 test may only use the 1974–1994 calibration (−0.147 per pp, t −1.7). On P/Ê, 2003–25 is also negative (t −1.9), so it is open whether λ_π should be active only in the high-inflation regime.
5. The earnings-cycle anchor needs two coefficients, κ_e and κ_m, fitted to train_end. Both legs are predictable but the price is not. How the engine applies the pair over a 10-year scenario is not implemented.
6. Reported EPS includes write-downs, which distort P/Ê in 2001–02 and 2008–09. The bridge items (foreign profits, sector mix, write-downs) have no cached series. NIPA rest-of-world profits could be added from the BEA flat files.
7. The Z.1 total net payout yield covers all nonfinancial corporations, not CRSP constituents. Its quarterly buyback data are noisy, which is why a 20-quarter mean is used.
8. The training rule for the cause label (same sign as the Kilian index) can disagree with the narrative; 1986Q1 is labelled demand. Over F the label is only marginal (lower bound 0.000).
9. V-AGG is 0.020 below the direct equation in 2005–14 (90% interval [−0.036, +0.003]). The test must be redone inside the joint model before switching.
10. No allowed source measures interest on debt held by the public. The identity uses the trailing 20-quarter mean of the 10-year yield, which understates interest when bills reprice fast (2023–24).
11. Announced paths (timing class 预告路径) are not a first-version training input. Their out-of-sample gain is unstable and negative over F [critic k01].
12. The tiers were still selected using all three test decades, so they are not out of sample. Quarters after the freeze (2026Q4 on) are the only clean holdout.

## Files the role listed

Paths are relative to `scratchpad/sim_redesign/`. Only the files marked are in this archive; the rest (scripts, work outputs, data) stay in the research folder.

- [`schema/SCHEMA.md`](../schema/SCHEMA.md) (in this archive)
- `schema/schema.json`
- `schema/coverage_final.csv`
- `schema-revise-2/scripts/reg_fields_v3.py`
- `schema-revise-2/scripts/reg_assets_v3.py`
- `schema-revise-2/scripts/reg_rules_v3.py`
- `schema-revise-2/scripts/reg_meta_v3.py`
- `schema-revise-2/scripts/r09b_build.py`
- `schema-revise-2/scripts/p01_patch_build.py`
- `schema-revise-2/scripts/p02_patch_build.py`
- `schema-revise-2/scripts/v01_policy_shocks.py`
- `schema-revise-2/scripts/v02_cause_split.py`
- `schema-revise-2/scripts/v03_equity_identity.py`
- `schema-revise-2/scripts/v04_sector_aggregation.py`
- `schema-revise-2/scripts/v05_inflation_leg.py`
- `schema-revise-2/scripts/v06_ecyc_price.py`
- `schema-revise-2/scripts/d01_fred_dl.py`
- `schema-revise-2/work/v01_policy_shocks.json`
- `schema-revise-2/work/v02_cause_split.json`
- `schema-revise-2/work/v03_equity_identity.json`
- `schema-revise-2/work/v04_sector_aggregation.json`
- `schema-revise-2/work/v05_inflation_leg.json`
- `schema-revise-2/work/v06_ecyc_price.json`
- `schema-revise-2/work/schema_compact.md`
- `schema-revise-2/work/decomp_compact.md`
- `schema-revise-2/work/coverage_compact.md`
- `schema-revise-2/work/schema_short.md`
- `schema-revise-2/work/r09_build.log`
- `schema-revise-2/dl/fred_20261101/FYOIGDA188S.json`
- `schema-revise-2/dl/fred_20261101/_requests_log.json`
- `schema-revise-2/backup_prev/SCHEMA_revise1_v2.md`
- `schema-revise-2/backup_prev/schema_revise1_v2.json`
- `schema-revise-2/backup_prev/coverage_final_revise1_v2.csv`
