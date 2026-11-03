# WorldSim: a small local world simulation

WorldSim is a separate, self-contained app in this repository (`worldsim/`). It does **world simulation only**:
no asset pricing, no charts, nothing from Causeway's engine at run time.

```
input state (schema values + text) ──► every actor decides (in parallel) ──► the synthesizer resolves all actions
        ▲                                                                              │
        └──────────────── output state: the same schema + text (one click = one step) ◄┘
```

- **State** = a value for every field of a typed **schema** plus **text**: a narrative, key questions (open /
  answered, with the current answer) and attributes (name → value).
- **Actors** have a role, what they care about, an **action space** (plus free actions if allowed), private
  information only they see, the fields they can see, and an optional model.
- The **synthesizer** sees the current state, the schema, every actor's output and its instructions, and writes the
  next state in the same shape, with a reason for every changed value.
- LLM calls go to **OpenRouter** with your key (default model `google/gemini-4-argon`), or to a deterministic **mock**
  that needs no key.

## Run it

```bash
pip install -e ".[dev]"              # once (fastapi, uvicorn, requests; the same install as Causeway)
python -m worldsim                   # http://127.0.0.1:8940
WORLDSIM_PORT=8950 python -m worldsim
WORLDSIM_LLM=mock python -m worldsim # force the free mock backend (no key, no network)
```

Open http://127.0.0.1:8940. On the first start the server creates the preloaded world 「美国与世界 · 2026Q4」.
Click **Run one iteration** on the Simulation page: each click is one step (one quarter in the default world).

| Environment variable | Default | What it does |
|---|---|---|
| `WORLDSIM_PORT` | 8940 | the port (the server always binds 127.0.0.1) |
| `WORLDSIM_DATA_DIR` | `data/worldsim` | where worlds, settings and the model-list cache live (git-ignored) |
| `WORLDSIM_LLM` | — | `mock` or `openrouter`: overrides the backend chosen in Settings |
| `WORLDSIM_OFFLINE` | — | `1`: never fetch the public model list (only the default model is listed) |
| `WORLDSIM_MOCK_DELAY_S` | 0.3 | how long each mock call takes, so the live progress is visible |
| `WORLDSIM_MOCK_SCRIPT` | — | scripted mock failures to try the failure paths, e.g. `{"synth": ["error"]}` (the next synthesizer call fails; behaviours: `ok`, `error`, `timeout`, `bad_json`, `invalid`; keys `synth`, `actor:<id>`, with `:repair` for repair calls) |

## Where to put the OpenRouter key

Open **Settings → OpenRouter API key**, paste the key (`sk-or-v1-…`, from https://openrouter.ai/keys) and click
**Save key**; **Test key** calls `GET https://openrouter.ai/api/v1/key` and shows the usage and limit.

- The key is stored only in `data/worldsim/settings.json` (git-ignored), written with file mode **0600**.
- The browser never gets it back: Settings shows it masked (`sk-or-…abcd`). It can be replaced or cleared.
- It is sent only to `openrouter.ai` in the `Authorization` header. It is never written to logs, error messages,
  prompts, saved iterations or exports (errors are redacted; a test checks every saved file).
- WorldSim does not read `.env` or an environment variable for the key.
- Without a key, choose **Mock (no key)** in Settings (or start with `WORLDSIM_LLM=mock`) to try everything for free.

## Models and cost

- **Default model**: `google/gemini-4-argon` (in OpenRouter's public model list on 2026-11-02: $2 / M input tokens,
  $10 / M output tokens, 1,048,576 context, structured outputs, reasoning always on with efforts low / medium / high).
- **Model picker**: the public list (`GET https://openrouter.ai/api/v1/models`, no key, cached for a day in
  `data/worldsim/models_cache.json`) with search; any model id can also be typed. Overrides: per world (Worlds page),
  per actor (Actors page) and for the synthesizer (Synthesizer page).
- **Settings**: temperature, reasoning effort (blank = the model's default), max parallel calls (default 6), timeout
  per call (default 240 s), max tokens (blank = provider default).
- **Requests**: `POST /api/v1/chat/completions` with `HTTP-Referer: http://localhost`, `X-Title: WorldSim`,
  `usage: {include: true}`, and `response_format: {type: "json_schema", strict: true}` when the model lists structured
  outputs (with `provider.require_parameters`), else `{type: "json_object"}`; if a provider refuses the JSON schema
  (HTTP 400/404/422) the call is repeated once in JSON mode. Every answer also goes through a tolerant JSON parser.
- **Cost**: the Simulation page shows an estimate before each click (the real prompts' length, an assumed answer
  length and a reasoning allowance, priced with the model list) and the actual cost after it (OpenRouter's
  `usage.cost`; computed from the list prices when absent). The default world's estimate with Gemini 4 Argon is about
  **$0.65–0.75 per iteration** (14 actor calls + 1 synthesizer call). Mock runs are free.

## An iteration, step by step

1. Every **enabled actor** is called in parallel (at most *max parallel* at once). Its prompt holds the world
   description, its role, what it cares about, its action space, its private information, the current state (only
   its visible fields) and the last *history window* steps (summaries, narratives, everyone's public actions and
   statements). It answers one JSON object:
   `{action_id, action_detail, rationale, public_statement, expected_consequences[]}`.
2. The **synthesizer** gets the world description, its instructions, the full schema (types, units, options,
   plausible ranges, historical-range notes, descriptions), the current state, the history and every actor's output
   (rationales included; private information is not). It answers
   `{summary, values: {field: {value, reason}}, narrative, key_questions[], attributes[]}`.
3. **Validation** against the schema: numbers / percents / integers / booleans / enums are coerced when the text is
   readable (noted), else the previous value is carried forward and flagged; a missing or null field is carried
   forward and flagged; a value outside the plausible range is kept and flagged; a changed value without a reason is
   flagged; unknown fields are ignored and flagged. Nothing is invented.
4. The new state is saved as the next step, with the iteration record: every actor's output, the synthesizer's
   answer, the flags, the changes, and **every call's prompt, raw response, token usage and cost** (Simulation →
   *Show prompts & raw responses*).

**Failures.** An actor call that fails is retried once, then the actor is marked *no response* and the synthesizer
is told to treat it as the status quo. An unusable answer (bad JSON, an action outside the space when free actions
are off) gets one repair call first. A synthesizer failure leaves the state unchanged: the error is shown with a
**Retry synthesizer** button (the actors' outputs are kept, so only the synthesizer is called again) or **Discard**.

**History.** Every step can be viewed with the iteration that produced it. **Rewind** to a step drops the later steps
(after a confirmation); **Fork** creates a new world with the history up to that step.

## The pages

- **Simulation**: the state as a grouped table (field, value, change from the previous step, the reason, flags),
  the narrative, key questions (new / answered badges), attributes (changed ones marked), *Run one iteration* with
  live per-actor progress, cost estimate and actual cost, the step history, the step viewer, rewind and fork.
- **Actors**: list and editor (name, id, enabled, role, cares, action space rows with reorder, free actions, private
  info, visible fields, model override).
- **Schema**: fields (add, edit, delete, reorder, group, type, unit, description, options, plausible min/max,
  historical-range note, tag) with the initial values, plus the initial period, narrative, key questions and
  attributes. The initial state is editable while the world has no later steps (rewind to step 0 or fork otherwise);
  fields can be added at any time (the synthesizer fills a new one).
- **Synthesizer**: instructions and model override.
- **Worlds**: the current world's name, description, output language, period step (quarter / month / year × n),
  history window and model settings; the list (open, duplicate, export JSON, delete with confirmation); create blank
  or from the template; import a JSON export (history included).
- **Settings**: backend, OpenRouter key, default model, call parameters, the model list.

The UI is plain ES modules (`worldsim/web/`), no build step. All server and model text is inserted as text, never
as HTML, under a strict Content-Security-Policy. It works at 1,100 px and at 375 px.

## Storage

```
data/worldsim/settings.json                    backend, key (0600), default model, call parameters
data/worldsim/models_cache.json                the public model list (refreshed daily)
data/worldsim/worlds/<id>/world.json           configuration: schema, actors, synthesizer, initial state
data/worldsim/worlds/<id>/steps/000000.json    the state at each step (step 0 = the initial state)
data/worldsim/worlds/<id>/iterations/000001.json   how step n was produced (outputs, flags, prompts, usage, cost)
data/worldsim/worlds/<id>/pending.json         a failed synthesizer attempt awaiting retry
```

## Security

The server binds `127.0.0.1`. Every `/api` request must carry a local `Host` header (no DNS rebinding); every
state-changing request (POST / PUT / PATCH / DELETE) must come from the app's own pages (`Origin` /
`Sec-Fetch-Site`, as Causeway's agents guard) and carry a JSON body. Deleting a world and rewinding need
`{"confirm": true}`.

## The default world: 「美国与世界 · 2026Q4」

Built by `tools/build_worldsim_default.py` from Causeway's local data and committed as
`worldsim/defaults/us_2026q4.json`, so it loads without `data/` (CI). Re-run the script after a data refresh;
`--check` reports whether the committed file is current (a test runs it when `data/` exists).

- **Values**: the state at the start of 2026Q4. The panels (`data/processed/panel_monthly.csv`, `panel_quarterly.csv`)
  end 2026-09; raw FRED series fill payrolls, housing starts, the mortgage rate, the breakeven and wages; the actors'
  books (`data/results/agents_accounts_root.json`) give the EFFR, the Fed's balance sheet, the fiscal lines, gasoline,
  the SPR and the yuan (their dates are in each field's description). Each numeric field has its historical range
  (from the same series) and plausible bounds (the historical range widened by a quarter of its span; judgment
  bounds where no local history exists, marked so).
- **The numbers quoted when the world was specified all match the data**: 10-year TIPS 2.93%, debt held by the
  public / GDP 97.37%, consumer sentiment 48.1 (this is the University of Michigan index, not the Conference
  Board's), initial claims 200.5k (September monthly average), labour share index 93.4, after-tax profits / GDP
  13.14%, dividend yield 1.03%, IT investment / GDP 5.05%. Noted in descriptions: the panel's effective tariff rate
  8.22% is 2026Q2, the start value 6.85% is the NIPA 2026Q3 rate (refunds of $134.4bn since May, cause unconfirmed);
  the 10-year yield was 5.29% at the end of September and 4.98% at the end of October.
- **Qualitative fields** come from the pinned facts in `data/agents/institutions.json`; every one still unconfirmed
  there is tagged 未确认 (two judgment fields are tagged 判断).
- **Text**: a five-paragraph Chinese narrative, eight key questions (the Fed's path with inflation above 3%, the
  midterms, the IEEPA tariffs, the oil disruption, the AI capex cycle, the fiscal path, the China truce, the labour
  market) and nine attributes.
- **Actors**: fourteen. Roles and action spaces of the ten policy / state / corporate actors come from Causeway's
  catalogue (`scenariolab/agents/actors.py`, its option labels, with a few made self-explanatory, e.g. 「维持 IEEPA
  关税授权」); the four private-sector actors from the state-synthesis design note (§7). Private information is each
  actor's book (balance sheet, income, cash flow highlights) plus the pinned facts that concern it.
- **Synthesizer instructions** (Chinese): resolve all actions jointly; keep fields consistent with each other and with
  the actions; change only what the quarter's events move; a reason for every change; plausibility against history;
  honest key-question updates; no asset price levels.

| Group | Field (uid=501(dongyujia) gid=20(staff) groups=20(staff),12(everyone),61(localaccounts),79(_appserverusr),80(admin),81(_appserveradm),98(_lpadmin),701(com.apple.sharepoint.group.1),702(com.apple.sharepoint.group.2),33(_appstore),100(_lpoperator),204(_developer),250(_analyticsusers),395(com.apple.access_ftp),398(com.apple.access_screensharing),399(com.apple.access_ssh),400(com.apple.access_remote_ae)) | Start value | Unit | Plausible range | Tag |
|---|---|---|---|---|---|
| 政策利率与收益率曲线 | 联邦基金利率（有效） (`policy_rate`) | 3.88 | % | 0 … 24 |  |
|  | 美联储总资产 (`fed_assets`) | 6745.6 | 十亿美元 | 0 … 11000 |  |
|  | 2 年期国债收益率 (`y2`) | 4.88 | % | 0 … 21 |  |
|  | 10 年期国债收益率 (`y10`) | 5.29 | % | 0 … 20 |  |
|  | 10 年期 TIPS 实际收益率 (`tips10`) | 2.93 | % | -2.3 … 4.3 |  |
|  | 10 年期限溢价（ACM） (`term_premium`) | 0.89 | 百分点 | -3.1 … 6.9 |  |
| 物价与通胀预期 | CPI 同比 (`cpi_yoy`) | 3.72 | % 同比 | -8 … 19 |  |
|  | 核心 CPI 同比 (`core_cpi_yoy`) | 2.57 | % 同比 | -3 … 17 |  |
|  | 核心 PCE 同比 (`core_pce_yoy`) | 3.12 | % 同比 | -2 … 12 |  |
|  | 1 年期通胀预期（密歇根） (`infl_exp_1y`) | 4.6 | % | -3 … 13 |  |
|  | 长期通胀预期（密歇根 5–10 年） (`infl_exp_lr`) | 3.4 | % | 1.5 … 5.4 |  |
|  | 10 年期盈亏平衡通胀率 (`breakeven10`) | 2.35 | % | -0.5 … 3.6 |  |
| 增长与投资 | 实际 GDP 增长（季度年化） (`real_gdp_growth`) | 3.3 | % 年化 | -49 … 46 |  |
|  | IT 设备与软件投资/GDP (`it_invest_gdp`) | 5.05 | % GDP | 0 … 6.2 |  |
|  | 超大型科技公司资本开支（近 4 季合计） (`hyperscaler_capex_ttm`) | 691.0 | 十亿美元 | 0 … 2000 |  |
| 劳动力市场 | 失业率 (`unemployment`) | 4.2 | % | 0 … 18 |  |
|  | 非农就业月均新增（近 3 个月） (`payrolls_3m`) | 50.7 | 千人/月 | -9800 … 5500 |  |
|  | 首次申领失业救济（月均） (`initial_claims`) | 200.5 | 千人/周 | 0 … 5800 |  |
|  | 时薪同比（生产与非管理人员） (`wage_growth`) | 3.3 | % 同比 | -1 … 12 |  |
|  | 劳动收入占比（指数，2017=100） (`labor_share`) | 93.4 | 指数 | 80 … 130 |  |
| 信用与金融条件 | Baa 信用利差（相对 10 年国债） (`baa_spread`) | 1.47 | 百分点 | 0 … 7.6 |  |
|  | GZ 信用利差 (`gz_spread`) | 0.93 | 百分点 | 0 … 9.7 |  |
|  | 银行放贷标准（SLOOS 净收紧比例） (`lending_std`) | 0.0 | % 净收紧 | -100 … 100 |  |
| 财政 | 联邦赤字/GDP (`deficit_gdp`) | 5.22 | % GDP | -12 … 35 |  |
|  | 联邦公众持有债务/GDP (`debt_gdp`) | 97.37 | % GDP | 0 … 130 |  |
|  | 净利息支出/联邦收入 (`net_interest_share`) | 19.46 | % | 0 … 50 |  |
|  | 债务上限余量 (`debt_limit_headroom`) | 713.2 | 十亿美元 | 0 … 10000 |  |
| 住房 | 新屋开工（年化） (`housing_starts`) | 1221.0 | 千套 | 0 … 3000 |  |
|  | 房价同比（Freddie Mac） (`hpi_yoy`) | 2.19 | % 同比 | -21 … 27 |  |
|  | 30 年期抵押贷款利率 (`mortgage30`) | 7.03 | % | 0 … 23 |  |
| 企业与收入分配 | 企业税后利润/GDP (`profit_share`) | 13.14 | % GDP | 1 … 16 |  |
|  | 美股股息率 (`dividend_yield`) | 1.03 | % | 0 … 8.7 |  |
| 居民 | 个人储蓄率 (`saving_rate`) | 3.8 | % 可支配收入 | 0 … 40 |  |
|  | 家庭偿债负担 (`hh_debt_service`) | 11.11 | % 可支配收入 | 7 … 18 |  |
|  | 消费者信心（密歇根） (`consumer_sentiment`) | 48.1 | 指数（1966Q1=100） | 20 … 130 |  |
| 外部、能源与贸易 | 贸易加权美元指数（广义） (`usd_index`) | 121.0 | 指数 | 60 … 200 |  |
|  | WTI 原油价格 (`wti`) | 97.2 | 美元/桶 | 0 … 180 |  |
|  | 汽油零售价 (`gasoline`) | 4.27 | 美元/加仑 | 1 … 8 |  |
|  | 战略石油储备库存 (`spr`) | 282.6 | 百万桶 | 0 … 750 |  |
|  | 有效关税率 (`tariff_rate`) | 6.85 | % | 0 … 14 |  |
|  | 人民币兑美元 (`cny_usd`) | 6.67 | 元/美元 | 5.5 … 8.5 |  |
| 信心与不确定性 | 经济政策不确定性指数 (`epu`) | 207.6 | 指数 | 0 … 910 |  |
|  | 地缘政治风险指数 (`gpr`) | 136.8 | 指数 | 0 … 380 |  |
| 政治与制度（定性） | 美联储政策立场 (`fed_stance`) | 偏鹰观望（暂停加息） | enum | — | 判断 |
|  | 最近一次议息会议（10 月 27–28 日） (`fomc_last`) | 维持 | enum | — | 未确认 |
|  | 美联储主席与理事会人事 (`fed_chair`) | 按数据日期的现任；主席倾向中性 | text | — | 未确认 |
|  | IEEPA 关税的法律状态 (`ieepa_status`) | 有效 | enum | — | 未确认 |
|  | 石油供给中断（2026 年冲突） (`oil_disruption`) | 中断持续 | enum | — | 未确认 |
|  | 两院与白宫的多数党 (`political_control`) | 白宫：共和党；参议院：共和党 53–47；众议院：共和党微弱多数 | text | — | 未确认 |
|  | 众议院控制权（2026-11-03 中期选举） (`house_control`) | 选举前：共和党微弱多数（结果未揭晓） | enum | — | 未确认 |
|  | 中美贸易休战 (`china_truce`) | 休战有效（约 2026-11-10 到期） | enum | — | 未确认 |
|  | 欧佩克+ 产量政策 (`opec_policy`) | 维持配额 | enum | — | 未确认 |
|  | 债务上限状态 (`debt_limit_status`) | 接近上限 | enum | — | 判断 |
|  | 对华芯片出口管制与稀土管制 (`chip_controls`) | 按数据日期的现状 | text | — | 未确认 |

| Actor (uid=501(dongyujia) gid=20(staff) groups=20(staff),12(everyone),61(localaccounts),79(_appserverusr),80(admin),81(_appserveradm),98(_lpadmin),701(com.apple.sharepoint.group.1),702(com.apple.sharepoint.group.2),33(_appstore),100(_lpoperator),204(_developer),250(_analyticsusers),395(com.apple.access_ftp),398(com.apple.access_screensharing),399(com.apple.access_ssh),400(com.apple.access_remote_ae)) | Action space (status quo first; free actions allowed) |
|---|---|
| 美联储 (`fed`) | 按兵不动 · 加息 25bp · 加息 50bp · 降息 25bp · 降息 50bp · 恢复缩表 · 准备金管理购买（扩大） |
| 总统与白宫 (`president`) | 维持现有关税 · 行业性加征 · 广泛加征 · 降级或豁免 · 达成协议（下调并承诺） · 动用战略石油储备 · 公开向美联储施压 |
| 国会 (`congress`) | 无新立法（只有自动稳定器） · 提高债务上限 · 暂停债务上限 · 债务上限对峙（未解决） · 小规模刺激（退税/直接支付） · 财政整顿 · 政府停摆 |
| 财政部 (`treasury`) | 发行结构不变 · 提高短债占比 · 拉长发行期限 · 扩大回购 · 动用非常规措施 |
| 最高法院与法院 (`courts`) | 本季不作相关裁决 · 维持 IEEPA 关税授权 · IEEPA 授权无效（不退税） · IEEPA 授权无效并退税 · 允许罢免美联储理事 · 反垄断：结构性救济（拆分/剥离） |
| 选民 (`voters`) | 执政党保住众议院 · 执政党失去众议院 · 执政党失去两院 |
| 中国 (`china`) | 不回应 · 对等加征关税 · 加征关税并收紧出口管制（稀土等） · 谈判或休战 · 大规模刺激（财政/信贷） · 人民币有管理的贬值 · 台海大规模军演 |
| 欧佩克+ (`opec`) | 维持配额 · 增产 · 减产 |
| 冲突方 (`belligerents`) | 僵持 · 升级 · 停火或航道恢复 |
| 超大型科技公司 CEO (`bigtech`) | 资本开支：维持或小幅调整 · 资本开支：大幅上调 · 资本开支：下调（消化） · 资本开支：大幅削减（2001–02 型） · 冻结招聘（AI 替代白领） · 削减回购为资本开支融资 · AI 价格战 |
| 消费者/居民 (`households`) | 按收入正常消费 · 预防性储蓄 · 动用储蓄维持消费 · 去杠杆 · 住房：观望 · 住房：抢先入市 |
| 华尔街投资者 (`investors`) | 维持配置 · 加仓风险资产 · 降低风险 · 避险抛售 · 科技仓位：加仓 · 科技仓位：轮动出去 |
| 银行/放贷机构 (`banks`) | 标准不变 · 放松 · 收紧 · 大幅收紧 |
| 非科技企业 (`firms`) | 按计划经营 · 扩张 · 放缓 · 收缩 · 定价：提价转嫁 · 定价：让价保份额 |

## API (JSON, under `/api`)

| Method and path | What it does |
|---|---|
| `GET /health`, `GET /settings`, `PUT /settings`, `POST /settings/test_key` | status; settings (key masked); update (`api_key`, `clear_key`, `backend`, `default_model`, `temperature`, `reasoning_effort`, `max_parallel`, `timeout_s`, `max_tokens`); check the key |
| `GET /models?refresh=1` | the model list |
| `GET /worlds`, `POST /worlds` (`{mode: blank\|template, name}`), `POST /worlds/import` | list, create, import |
| `GET\|PUT\|DELETE /worlds/{id}` | read (with steps and a pending failure), update settings, delete (`{confirm: true}`) |
| `PUT /worlds/{id}/schema` (`{fields, initial_values?}`), `/initial`, `/actors`, `/synthesizer` | edit (validated; 400 with `problems`) |
| `POST /worlds/{id}/duplicate`, `GET /worlds/{id}/export` | copy without history; download with history |
| `POST /worlds/{id}/run`, `GET /worlds/{id}/run`, `POST /worlds/{id}/retry`, `POST /worlds/{id}/discard` | start an iteration (background), live progress, retry or drop a failed synthesizer attempt |
| `GET /worlds/{id}/steps`, `/steps/{n}`, `/iterations/{n}/calls`, `/pending/calls` | history, one step with its iteration, the prompts and raw responses |
| `POST /worlds/{id}/rewind` (`{step, confirm: true}`), `POST /worlds/{id}/fork` (`{step, name}`) | rewind, fork |
| `GET /worlds/{id}/estimate` | the cost estimate of the next iteration |

## Tests

`tests/test_worldsim_*.py` (with the fake OpenRouter in `tests/worldsim_fakes.py`) run on the mock backend or a fake
HTTP transport; `tests/conftest.py` points WorldSim at a temporary data directory and replaces its transport with
one that refuses every request, so no test can reach openrouter.ai. They cover world CRUD, schema and state
validation, an iteration end to end (actors in parallel, synthesizer, validation, persistence), the failure paths
(actor retry and no response, bad-JSON repair, synthesizer failure and retry), the OpenRouter request shape, the
key (0600, masked, absent from every other saved file, response and log), the origin guard, rewind and fork,
export / import, and the default world loading without `data/`.

## Limits

- One iteration at a time per world; progress lives in the server's memory (a restart during a run loses the run,
  never a saved step). Runs of different worlds can overlap.
- The cost estimate is rough (token counts estimated from text length; answer lengths and reasoning assumed).
- The mock backend only exercises the mechanics: its actions are picked by hash and its value changes are small
  deterministic moves, not economics.
- Prompts are fixed in English around the world's own text; output language is the world's setting.
- Numbers in the default world are a snapshot of Causeway's local data on 2026-11-02; most pinned facts are
  unconfirmed and tagged so.
