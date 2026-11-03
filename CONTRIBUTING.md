# 参与贡献 · Contributing

欢迎提 issue 和 pull request。这份说明讲怎样搭环境、构建、跑测试，以及添加情景和数据源时要遵守的约定。模块之间的契约（每个 JSON 字段、每个构建步骤、每个 API）写在 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：改动其中任何一个形状，就要同时改所有的生产者、使用者和这份文档。

*English summary at the end.*

## 环境

需要 Python 3.10 或更新的版本。

```bash
git clone https://github.com/Dongyu-Jia/causeway.git
cd causeway
python3 -m venv .venv && source .venv/bin/activate      # Windows：.venv\Scripts\activate
pip install -e ".[llm,dev]"                             # llm 只在 MSL_LLM_BACKEND=api 时需要（默认用本机 Claude Code），dev 是测试和 ruff
```

- 网页没有构建步骤：`web/` 是静态文件（ES 模块），由 `python -m scenariolab.server` 直接提供。
- 图表库 ECharts 5.5.1 已随代码放在 `web/vendor/echarts/`（Apache License 2.0，同目录有 LICENSE 和 NOTICE），不依赖 CDN。升级时从 npm 注册表下载对应版本的 tgz，核对注册表给出的 `dist.integrity`，只保留 `dist/echarts.min.js`、`LICENSE` 和 `NOTICE`，再改 `web/index.html` 里的版本注释和 README。
- LLM 翻译默认用本机 Claude Code 和你自己的订阅登录（`claude auth login`），不需要 API key，只供个人在本机使用；只有 `MSL_LLM_BACKEND=api` 时才在 `.env` 里填 `ANTHROPIC_API_KEY`（`cp .env.example .env`）。**永远不要提交 `.env`**。
- FRED 的 API key（免费）是可选的：在 `.env` 里填 `FRED_API_KEY=` 后，构建通过 `api.stlouisfed.org` 下载 FRED 序列，否则用 FRED 的图表 CSV；key 被拒绝或 API 不可用时也自动改用图表 CSV。key 只作为 `api_key` 参数发给 `api.stlouisfed.org`，不写进任何文件名、缓存、日志或结果文件；日志和 `data/raw/SOURCES.json` 里的网址一律写成 `api_key=REDACTED`。

## 构建

```bash
python -m scenariolab.build                 # 首次运行先下载公开数据（无需 key），再估计模型、跑检验和情景
python -m scenariolab.build --skip panel    # 不重建数据面板，用已缓存的面板重建全部结果
python -m scenariolab.build --only scenarios regimes forecast valuation portfolio llm meta   # 改了内置情景之后
python -m scenariolab.build --only ablation # 默认不跑的引擎消融（约 5 分钟）
python -m scenariolab.build --only regime_forecast   # 默认不跑的宏观体制一年期预测检验（约 2 分钟）
python -m scenariolab.server                # http://127.0.0.1:8765
```

- 结果写进 `data/results/`，网页直接读这些文件。各步骤读什么、写什么见 ARCHITECTURE.md 的 “The build”。
- 构建之后重启服务器：自定义情景和 LLM 翻译用的是服务器启动时载入的引擎。
- 实验和对照放在 `data/results/` 之外（例如 `scenarios.library.run(ctx, out_dir=<临时目录>)`、`python -m scenariolab.analysis.ablation --scen-dir <目录>`），不要覆盖网页正在读的结果。

## 测试

```bash
pytest -q                            # 全部测试，完整构建之后约 9 分钟（没有数据时依赖数据的测试跳过，约 2 分钟）
ruff check scenariolab tests tools worldsim   # 规则见 pyproject.toml（E9 语法错误 + F pyflakes）
python -m compileall -q scenariolab worldsim
```

- 需要本地数据或结果文件的测试在构建之前会自动跳过；新克隆的仓库没有 `data/`，CI（`.github/workflows/ci.yml`，Python 3.10 和 3.12）就在这种状态下运行编译检查、ruff 和测试。
- 新功能要有测试。测试不下载数据、不调用模型：网络用假的 `requests.get`（见 `tests/test_build.py`），API 用假的客户端（见 `tests/test_translator.py`），Claude Code 用假的 `subprocess.run`（见 `tests/test_claude_code.py`；`tests/conftest.py` 让任何测试都启动不了真正的 claude，也不消耗订阅用量）。
- **测试从不花 token。** 凡是会调用大模型的路径（月度翻译器、季度情景表翻译器、智能体推演、叙事检验），测试里一律用模拟后端：`MSL_LLM_BACKEND=mock`，代码在 `scenariolab/llm/mock.py`。它不起进程、不联网，只读提示词和用户消息，给出确定的、符合格式的回答。
  - 边界情况用录好的夹具：坏 JSON、拒绝、超时、用量上限、未登录等，用 `MOCK.push(name, match=...)` 或 `MSL_LLM_MOCK_FIXTURE` 触发。
  - 季度情景表可以用 `mock.register_qtable(narrative, qtable)` 回放。
  - 不要在测试里把后端设成 `claude_code` 或 `api`，也不要为了测试去掉“先估算、再确认”的关卡：叙事检验在 pytest 下会拒绝真实后端。
  - 真实调用只能是用户自己发起的命令或按钮，先显示估算（调用次数、tokens、分钟数），确认之后才发出；构建步骤也永远不调用真实模型。
- 需要数据的测试用 `needs_panel`（或检查它要读的那个结果文件）在缺数据时跳过，而不是失败。

## 预注册记录：永远不改冻结的记录

季度模型的几个关卡都先写好协议再运行，记录是冻结的：

- T-A（资产层候选）、T-C（补全）、T-S（切换关）放在 `scenariolab/qmodel/preregistered/<日期>_T?.json`，由预注册那一轮的工具（`tools/run_preregistered.py`）写入，这两样随那一轮合并进来；
- T-N（叙事检验能说什么）放在 `scenariolab/evaluation/narrative/preregistered/2026-11-02_TN.json`。

规则：

1. **永远不改冻结的部分**：协议、候选、门槛、种子、输入文件的哈希都不改。改了，基于它的结论就作废：切换关会把这样的结果判为“过期”，什么也不切换。
2. **结果只追加，从不改写**：一次运行、一条状态或一条说明，只能追加到记录的 `results` 里（由运行它的工具写）。切换关的评估另外把每次结果追加到同名的 `<stem>.results.jsonl`。
3. **协议要改，就另立一份新的记录**：新记录有新的日期或标签（例如 `2026-11-02-r2_TC.json`），并写明它取代哪一份。
4. **先提交冻结的记录，再提交结果**：两次提交分开，不要把这几次提交 squash 合并。预注册那一轮合并之后，CI 会按完整的提交历史检查这个顺序。
5. **记下在哪些数据上选了什么**：2015–2026 年已经被前几轮看过，是重新检验，不是新鲜的留出样本。真正干净的检验只有从 2026Q4 起的前瞻账本。
6. **研究存档也冻结**：`docs/research/2026-11-02-sim-redesign/`、`scenariolab/evaluation/narrative/research/` 和 `scenariolab/evaluation/research/oracle_ablation.json` 都不改；新的一轮研究另建文件夹。

## 季度模型的约定

- 生成的文件不要手改，改了用对应的工具重新生成。测试和工具的 `--check` 会核对它们：
  - `scenariolab/qmodel/card.json`：`python tools/build_model_card.py`
  - `scenariolab/qmodel/schema_view.json`：`python tools/schema_view.py`
  - `scenariolab/scenarios/qtables/*.json`：`python tools/qtable_from_yaml.py`
- 叙事检验的钥匙 `data/evaluation/key.json`（哪个编号是哪段历史）只在本地，不进 git，也不要贴进 issue 或 PR。`prompts.py` 和 `runner.py` 永远不能导入 `key.py`，测试会检查导入关系；构造或发送提示词的代码里不要出现片段的年份或起点。
- 界面文字遵守 [SIM_DESIGN](docs/research/2026-11-02-sim-redesign/SIM_DESIGN.md) §10.6，有文字测试检查：
  - 「预测」只用于情景概率和前瞻账本；情景结果叫“情景条件模拟”或“推演”；
  - 12–20 个季度的数字一律标「只能当个案」；
  - 「估值假设」一律带「相当于直接假设股价」；
  - 没有通过的预注册声明，不能写成“有改进”；
  - 大模型的文字只用 `textContent` 或 `h()` 放进页面，不经过 `innerHTML`。

## 添加一个内置情景

1. 在 `scenariolab/scenarios/specs/` 下复制一个最接近的 YAML，文件名就是默认 id。字段和边界见 ARCHITECTURE.md 的 “Scenario spec”：视图只能用 `s_` / `f_` / `e_` / `rg_`（事后体制标签）变量，目标值会被截断到“该阶段 6 个月均值”的历史 2%–98% 分位，覆盖项（factor_drift、asset_drift、resid_scale、r_star）只用于历史上没有先例的部分，每一项都写 `why`。
2. `tests/test_scenarios.py` 检查的约定：name、summary、tags、analogs 不为空；narrative 4–8 句（按“。”计）；因果链 5–10 条边；2–4 个阶段、合计 180 个月；不得对资产收益设视图；校验时没有错误、截断和警告；全部内置情景的 prior_prob 合计在 0.95–1.05 之间（加一个情景时相应调低其他情景的先验）。
3. `calibration`（收入分配计算器）只有内置情景库会换算成视图和漂移；自定义情景和 LLM 规格里的 calibration 块会被忽略并给出警告。
4. 运行 `pytest tests/test_scenarios.py`，再重建：`python -m scenariolab.build --only scenarios regimes forecast valuation portfolio llm meta`（宏观体制页和情景的体制读数共用引擎的无条件参照，一年期预测、估值页、组合默认值和 LLM 的示例都依赖情景结果或情景文件）。
5. 阶段可以用 `regime` 字段要求多抽某种宏观体制的历史月份（例如 `regime: {滞胀: 0.7}`）；校验会给出历史支撑不够的警告（W1–W11），内置情景要做到没有这类警告。阶段名称或叙事写到的体制（滞胀、衰退、通缩、过热……）要在模拟路径的这一阶段里占一半以上的月份（体制读数的 N3 注记检查名称），先用视图做到（regime 字段、rg_ 标志、f_cpi 或情景变量）；历史撑不起来时（截断、支撑太薄、收缩比 1948 年以来最长的一次还长）就改名，并在叙事里写明实际的体制构成。W7 和 W11 只数抽中的收缩月份；要求多抽收缩类月份（收缩、衰退、深度衰退、滞胀）的阶段还要看模拟出来的收缩有多长：体制读数逐阶段给出与这一阶段重叠、长于 18 个月的收缩（`long_contraction`）和引擎无条件抽样同一时段的对照，收缩段里长于 18 个月的占比比对照高出 10 个百分点以上时出现 N5 注记——内置情景要做到没有 N5：把收缩阶段压到 12 个月左右、后面接一个复苏或常态阶段，或降低要求，并在叙事里写出这一占比和对照。“不衰退”这类说法要写出出现收缩的路径占比，并和引擎无条件抽样、同样长度的历史窗口对照。
6. narrative 里引用的模拟数字（“模拟的……约……”）来自引擎的运行结果，跑完之后再填写；改了视图、传导模型（走步检验换了模型）或数据面板之后都要重新核对，摘要（summary）和因果链里的数字也一样：模拟路径只由视图、种子和引擎决定，改 narrative 的文字不改变任何模拟数字。
7. 季度模型读的是同一个情景的季度情景表（`scenariolab/scenarios/qtables/<id>.json`）：
   - 改了 YAML 之后运行 `python tools/qtable_from_yaml.py`，逐条看换算记录（`conversion`）和人工复核记录（`review`），金融条件的立场要有原文依据；
   - 再运行 `python tools/qtable_from_yaml.py --check` 和 `--delivery`：每个视图都要落在月度引擎自己的交付容差之内；
   - YAML 的 sha256 记在表里，旧表会被测试发现。

## 添加一个数据源

1. 在 `scenariolab/data/sources.py` 里写下载函数，统一用 `_download()`：文件缓存在 `data/raw/<来源>/`，缺少时下载，已缓存的只在 `--refresh` 时重新下载；不需要 API key 的公开来源才能加。可选来源要“失败也不中断构建”（参照 `shiller()`：返回 `None`，并校验下载内容的格式，不把错误页面缓存下来）。
2. 在 `scenariolab/data/panel.py` 的 `build_panel()` 里把它变成面板的列（`s_` 水平、`f_` 月度变化、`r_` / `p_` / `d_` 资产回报、`lvl_` 原始水平），把来源加进 `SOURCES`（名称 → 网址，“方法与数据”页会列出），口径写进 `meta["notes"]`。新的因子或状态 id 加到 `config.FACTORS` / `config.STATES`。
3. 测试里用假的网络响应，不要真的下载（见 `tests/test_build.py`）。
4. 遵守数据来源自己的使用条款；需要付费或绕过访问限制的来源不加。唯一需要（免费）注册的是可选的 FRED API key，没有它构建照常用图表 CSV。
5. 下载的文件是不可信的数据：只解析，不执行；Excel 文件用 xlrd / openpyxl 只读打开，不运行宏。每个来源先检查内容格式（文件头、魔数），把错误页面挡在缓存之外。

## 添加一个因子（缺失因子审计之后的规则）

模型的因子只通过走步检验进入，没有例外。加一个因子要同时做到：

1. **`config.FACTORS` 里一条带角色的记录**：`core`（默认的传导模型，改它会改变走步检验的固定对照）、候选（`core: False`）、`interaction`（交互项，写明状态变量、滞后和门槛）、`dedicated`（只进某个资产的方程）或 `role: "audit"`（只用于审计对照：不进模拟、视图、漂移和变量表）。情景视图变量加到 `config.STATES`，带 `concept` 和 `source`，晚开始的序列加 `late: True`。
2. **面板列和 `column_info`**：在 `data/panel.py` 里生成 `f_` / `s_` 列，并在 `panel.COLUMN_INFO` 里登记来源、对齐方式（参考月 / 按发布月 / 月末 / 月均 / 每月最后一周）和是否修订；季度数据一律按发布月对齐（发布滞后写进 `config.RELEASE_LAGS`），不插值。只给审计用的旧口径列用 `x_` 前缀，它们永远不能作为视图或模型输入。
3. **来源记录**：下载函数和缓存见上一节；`data/raw/SOURCES.json` 里有一条记录（网址、发布者、访问日期、条款、频率、首月、发布滞后、是否修订、版权）。
4. **审计登记**：在 `analysis/factor_audit.py` 的 `REGISTERED` 里登记候选（它的列、名称和研究存档里的对应项；复现模式按它重算），在 `analysis/walkforward.py` 的候选列表里预先登记（`build_candidates` 或 `DEDICATED_CANDIDATES`，写明是否可以被选中），然后运行 `python -m scenariolab.build --only eras diagnostics factor_audit`，再重建之后的全部步骤（eras 用选中的模型做传导归因、却排在 diagnostics 之前：两步同时运行时，走步检验换了模型，构建会在 diagnostics 之后自动重做 eras；只跑 `--only diagnostics` 时构建会提示补跑 `--only eras`）。
5. **研究存档冻结**：`scenariolab/analysis/audit_archive/research_2026_11_01.json` 和 `docs/research/2026-11-01-factor-audit/` 是 2026-11-01 研究轮的存档，不再修改。新的一轮研究加一个新的存档文件和一个新的 docs 文件夹。

## 宏观体制的判定参数

宏观体制分类器的参数只通过 `config.REGIMES` 修改，不在别处另写阈值；每次修改都要重跑 `python -m scenariolab.build --only scenarios regimes forecast valuation portfolio llm meta`：情景结果里的体制读数（每个情景的 `regimes`、情景库的第 1 年构成）由 scenarios 这一步按现行参数算出，宏观体制页（NBER 验证、样本外参数网格和数据修订研究都在 regimes 这一步里）和它们要用同一套参数；保留了一年期预测检验的，再跑 `--only regime_forecast`。在 PR 里报告 NBER 验证、样本外网格和数据修订研究这三项结果的变化。`high_inflation_pct` 与传导模型的 `factor_model.HIGH_INFLATION` 是两个常数，测试要求两者相等，改一个就要同时检查另一个。

## 代码风格

- Python 跟随周围代码的写法：类型标注，docstring 说明做什么和为什么；带证据的常数在模块 docstring 里写明依据；行宽约 120。代码、注释和 id 用英文；界面文字和 API 的错误信息用简体中文。
- 所有 JSON 都经过 `scenariolab.jsonio`（原子写入、NaN 变成 null、浮点数保留 6 位）。
- 前端：ES 模块、没有框架和构建步骤；图表通过 `web/js/charts.js`。颜色约定：`.pos` / `.neg` 表示数字的正负，跟随 K 线配色（默认红涨绿跌）；`.is-good` / `.is-bad` 表示好坏判断，不跟随。
- 不随意增加运行时依赖；需要的话先开 issue 讨论。

## 数据不进 git

`data/` 整个目录都在 `.gitignore` 里：下载的原始数据、派生的面板、构建结果、自定义情景和一年期预测的账本（`data/ledger/`，只追加、不覆盖，留在各自的电脑上）都不提交，也不要放进 issue 或 PR。各数据来源有自己的条款，仓库只分发代码。`.env`（令牌和 key）同样永远不提交；`data/llm_cache/`（翻译缓存）也在 `data/` 里。`docs/screenshots/` 里的界面截图可以提交（每张压缩到 300 KB 以内）。

## Pull request

- 一个 PR 做一件事；说明改了什么、怎样验证的（测试、重建了哪些步骤、在浏览器里看过哪些页面）。
- CI 必须通过。改了行为就同时更新 README 和 docs/ARCHITECTURE.md。

## English summary

- **Setup:** Python 3.10+, `pip install -e ".[llm,dev]"` (the `llm` extra is only for `MSL_LLM_BACKEND=api`; by default the translator runs the local Claude Code CLI on your own subscription, for personal local use only). The web UI is static ES modules with no build step; ECharts 5.5.1 (Apache-2.0) is vendored in `web/vendor/echarts/`.
- **Build:** `python -m scenariolab.build` downloads the public data on first run and writes `data/results/`; use `--skip panel` or `--only <steps>` to rebuild part of it, and restart the server afterwards. The module and JSON contracts are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Tests:** `pytest -q`, `ruff check scenariolab tests tools worldsim`. Tests never download data or call the API (use fakes); data-dependent tests skip without `data/`, which is how CI runs them on a fresh clone.
- **Tests never spend tokens:** every LLM path (the monthly translator, the QTable translator, the agent tree, the narrative test) runs on the mock backend in tests (`MSL_LLM_BACKEND=mock`, `scenariolab/llm/mock.py`: deterministic, schema-valid answers and recorded failure fixtures). Never point a test at `claude_code` or `api`, and never remove the estimate-and-confirm gate to make a test pass. A real call exists only as a command or button the user starts after an estimate.
- **Pre-registration:** never edit a frozen record (`scenariolab/qmodel/preregistered/`, which lands with the pre-registration round, and `scenariolab/evaluation/narrative/preregistered/`). Results are only appended; a protocol change is a new dated record that names the one it supersedes; commit the frozen record before its results and never squash those commits. The research archives (`docs/research/2026-11-02-sim-redesign/`, `scenariolab/evaluation/narrative/research/`, `scenariolab/evaluation/research/`) are frozen too.
- **Quarterly stack:** `card.json`, `schema_view.json` and `scenarios/qtables/*.json` are generated (`tools/build_model_card.py`, `tools/schema_view.py`, `tools/qtable_from_yaml.py`; `--check` compares). After editing a built-in YAML, regenerate and review its QTable. The narrative test's key (`data/evaluation/key.json`) stays local, and the prompt builder must never import `key.py`.
- **New scenario:** copy a YAML spec in `scenariolab/scenarios/specs/`, keep to the rules `tests/test_scenarios.py` checks, then rebuild `--only scenarios regimes forecast valuation portfolio llm meta`.
- **Regime classifier:** its parameters change only through `config.REGIMES`; every change re-runs `--only scenarios regimes forecast valuation portfolio llm meta` (the scenario step labels the scenario paths with the live parameters; the regimes step holds the NBER validation, the out-of-sample grid and the revision study), plus `--only regime_forecast` when that file is kept, and reports the three studies.
- **New data source:** a cached downloader in `scenariolab/data/sources.py` (through `_download()`; optional sources fail soft; check the content before caching, parse downloads and never execute them), panel columns and a `SOURCES` entry in `scenariolab/data/panel.py`, mocked network in tests. Open sources only; the one optional (free) key is `FRED_API_KEY`, which goes only to `api.stlouisfed.org` and is logged as `api_key=REDACTED`.
- **New factor:** a `config.FACTORS` entry with a role (core / candidate / interaction / dedicated / audit), a panel column registered in `panel.COLUMN_INFO` (quarterly data release dated with `config.RELEASE_LAGS`, never interpolated; `x_` for audit-only old definitions), a provenance record in `data/raw/SOURCES.json`, an audit registration in `analysis/factor_audit.py`, and a pre-registered walk-forward candidate. Rebuild from `--only diagnostics factor_audit` on. The research archive (`analysis/audit_archive/research_2026_11_01.json`, `docs/research/2026-11-01-factor-audit/`) is frozen: a new round adds a new archive file and a new docs folder.
- **Style:** follow the surrounding code; English code and comments, Simplified Chinese UI text; all JSON through `scenariolab.jsonio`.
- **No data in git:** `data/` and `.env` are git-ignored and must never be committed (`.env` may hold `ANTHROPIC_API_KEY` and `FRED_API_KEY`).
