# Episodes and narratives (research round, role `writer`)

*What the role did:* 12 historical windows with three Chinese narratives each (n1, n2, n3) and an anchored production prompt.

Rendered from the role's final report in the research workflow's journal (run `wf_cc7fe8c3-afc`, result `writer`). The text is unchanged except that local absolute paths are written as the spec writes them (`scratchpad/sim_redesign/...`, `scratchpad/...`). It is the report the critic and the designer read; the numbers are the role's own, and the critique ([`critic/critique.md`](../critic/critique.md)) and the spec ([`SIM_DESIGN.md`](../SIM_DESIGN.md)) say which of them survived.

## Summary

I wrote the inputs for the user's end-to-end test (narrative → blind LLM translation → factors → assets): 12 historical windows with 3 Chinese narratives each, plus an anchored production prompt per window. Everything is under scratchpad/sim_redesign/narratives/E01..E12, and nothing in MAIN, RWT or FWT was written.

**Windows.** The 9 required windows plus 3 calm or boom ones: 1995-07..97-12, 2004-07..06-12 and 2017-01..19-12. Codes were drawn with a random shuffle and are not chronological. The key exists only in this result: no file and no script contains a code-to-window mapping. The scripts take the window as a command-line argument.

**Narratives.** None names a date, year, person, country or event, and none says anything about asset prices, valuation, earnings or sectors. An automated scan for years, names and banned words is clean.
- N1 covers the policy rate, goods/services/core CPI, industrial production, unemployment, wages and labour share, the oil price, geopolitics, fiscal policy and the pandemic. Magnitudes and month-level timing are rounded from MAIN's panel (lvl_ff, s_cpi_yoy, s_goods_yoy, s_services_yoy, s_core_yoy, lvl_ip, lvl_unrate, lvl_oil, lvl_wage, lvl_labor_share). Facts outside the panel (fiscal sizes, events, recession dating) are qualitative or from the historical record.
- N2 is N1 verbatim plus a 金融条件 section (10-year yield, BAA−10Y spread, trade-weighted dollar, curve inversion) and an extended summary. This was checked programmatically.
- N3 is N1 rewritten with no numbers at all: no digits, no counts, no horizon length. It keeps only sequence words.

**prompt.md.** Built with translator.build_system_prompt (horizon 180) on an EngineContext whose model, MonthDB and anchor come from backtest.fit_window(panel, spec, T0). T0 is the month before the window and spec is model_selection 'base'. Three deviations from a plain call:
- The e_* structural-shock columns are dropped, because their SVAR is estimated on the full sample. The prompt then prints the builder's own 'no shocks' line, and validate_spec drops the oil_war example's e_oil_supply view.
- ctx.panel is panel.loc[:T0], so the k_shaped calibration base reads the 12 months up to T0.
- E06 cannot be anchored (only 8 model months precede it), so its prompt.md is leave-window-out: fitted on every model month except the window. E03 is anchored on only 80 months, and E03/prompt_lwo.md is a leave-window-out alternative.

The builder is scratchpad/sim_redesign/writer/w06_build_prompt.py. Its flags are --t0, --mode anchored|lwo, --exclude, --keep-shocks and --out; it logs every replacement to stdout and refuses to finish if any 19xx/20xx string remains.

**What was replaced in prompt.md.** The same rules apply to every code.
1. Month dates become offsets from T0, written T0−5m or T0−2y3m. This covers the sample line, 今天（T0）, the variables header, the lagged-date marks in the state table, era spans and example analogs. The month range in '1990 年 7-10 月' becomes 'T0−a 至 T0−b'.
2. Year references become Y0±k, where Y0 is T0's calendar year. '1973 年以来' (the sample start) is always kept. '1980 年利率 15%' becomes 'Y0−k 年…', '1970 年代' becomes 'Y0−k 年代', and ranges such as '1973-74 年' become 'Y0−a 至 Y0−b 年'. Year lists and '2008 年下半年' are handled the same way.
3. Two unit labels are tokenized: '实际油价(2026美元)' becomes '(Y0+k年美元)' and '2017=100' becomes 'Y0±k=100'.
4. To honour the no-look-ahead rule, any dated reference to something not finished by T0 is deleted together with the clause it dates. This is done silently, so the text does not hint at what follows. It covers:
   - In engine rule 3: '和 2021–2022 年' everywhere; '1970 年代' and the '1980 年利率 15%…幅度小得多；' clause for E03 and E06; and the whole example sentence when nothing is left of it.
   - In oil_war: '，2026 年 4 月同比一度超过 80%'; the items 1999/2009/2021; '1990 年海湾战争'; '1973-74 年禁运'; '1974-75/1990-91 年'; '（1990 年 7-10 月约 −25 点）'; '2008 年下半年…'; '1980-82 年'; and '…2020 年 4 月这类…封锁月份'.
   - In k_shaped: '（Moody's：2025 年约 49.7%）'; '（权重最高的是 2009-14 年的复苏月份）'; '2023 年以来约 +0.5'; and '，即不含 2020 年时的月度波动）排除 2020 年 4 月…'.
   - In both examples' analog lists, every analog that ends after T0.
   - From the era list, every config.ERAS entry not finished by T0, including the era that contains T0.
5. Nothing else was changed. The undated 2026-state sentences in the two examples, the AI factor_drift example and the summaries of finished eras were kept as they are.

**Sample size.** There are 12 non-overlapping episodes, 297 months (about 99 quarters) in all. E07 and E02 are adjacent.

The writer returned its code-to-window key only in its structured result, and the workflow kept only this summary: the key is not in this archive. The research episodes' windows are published in the spec (§8.2); the app's own narrative test uses new shuffled codes whose key is sealed in data/evaluation/key.json (scenariolab/evaluation/narrative/key.py).
