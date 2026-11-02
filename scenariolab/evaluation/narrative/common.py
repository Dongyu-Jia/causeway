"""Shared constants of the narrative test (SIM_DESIGN §8): paths, arms, horizons, asset groups, labels.

Nothing here knows an episode's window: the code-to-window key is read only by key.py (and through it by the
scorer, the audit and the seal command). The prompt builder (prompts.py) and the runner (runner.py) import this
module, episodes.py and the translator, never key.py; tests/test_narrative.py checks that.

Paths are module attributes read at call time (eval_dir(), runs_dir()), so a test can point the whole test at a
temporary folder by setting EVAL_DIR and RUNS_DIR.
"""
from __future__ import annotations

from pathlib import Path

from ... import config
from ...qmodel import config_q as cq

PACKAGE_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = PACKAGE_DIR / "research"              # the imported research round (committed; import-standin)
PREREG_DIR = PACKAGE_DIR / "preregistered"           # the frozen T-N record (claims, thresholds, seeds)
EVAL_DIR = config.DATA_DIR / "evaluation"            # the sealed registry: narratives, stand-ins, synthetic texts
RUNS_DIR = config.RESULTS_DIR / "narrative"          # <run_id>/ (manifest, translations, score, report)


def eval_dir() -> Path:
    return Path(EVAL_DIR)


def runs_dir() -> Path:
    return Path(RUNS_DIR)


def registry_path() -> Path:
    return eval_dir() / "registry.json"


def narratives_dir() -> Path:
    return eval_dir() / "narratives"


def standin_dir() -> Path:
    return eval_dir() / "standin"


def estimates_dir() -> Path:
    return runs_dir() / "estimates"


# ---------------------------------------------------------------- the registry
REGISTRY_SCHEMA = "narrative-registry/1"
NARRATIVE_VERSIONS = ("v1", "v2", "synth")           # v1: the research round's texts; v2: writer protocol v2 (to come)
NARRATIVES = ("n1", "n2", "n3")
SETS = ("clean", "short", "synth")
CODE_PREFIX = {"episode": "X", "synth": "S"}

# ---------------------------------------------------------------- arms (§8.3)
# kind: baseline / oracle (realized fields by endogeneity class) / identity / narrative (an LLM translation)
ARMS = {
    "U": {"name": "U", "zh": "无叙事（模型无条件模拟）", "kind": "baseline"},
    "OA": {"name": "O-A", "zh": "已知宏观：外生宏观、政策原始值与油价原因", "kind": "oracle", "classes": ("E0", "E0p")},
    "OAp": {"name": "O-A′", "zh": "O-A 加已知商品价格（“石油打仗”当作已知）", "kind": "oracle",
            "classes": ("E0", "E0p", "E1c")},
    "OB": {"name": "O-B", "zh": "O-A′ 加已知金融市场（利率、期限溢价、利差、美元、融资与杠杆、房地产价格）",
           "kind": "oracle", "classes": ("E0", "E0p", "E1c", "E1")},
    "OF": {"name": "O-full", "zh": "全部 x 字段已知（再加调查与新闻指数）：因子到资产这条链的上界", "kind": "oracle",
           "classes": ("E0", "E0p", "E1c", "E1", "E2")},
    "V": {"name": "V", "zh": "O-full 加已知估值倍数：恒等式上界（不是模型能力）", "kind": "identity"},
    "N1": {"name": "N1", "zh": "叙事 N1：宏观与政策（带大致幅度）", "kind": "narrative", "narrative": "n1"},
    "N2": {"name": "N2", "zh": "叙事 N2：N1 加金融条件", "kind": "narrative", "narrative": "n2"},
    "N3": {"name": "N3", "zh": "叙事 N3：不含数字", "kind": "narrative", "narrative": "n3"},
    "NW": {"name": "N-wrong", "zh": "错叙事：配对片段的 N1，按本片段的提示词翻译、从本片段的起点运行", "kind": "narrative",
           "narrative": "n1"},
    "NS": {"name": "N-synth", "zh": "合成片段：用模型自己的模拟路径写成的叙事（不可能被识别）", "kind": "narrative",
           "narrative": "n2"},
    "HS": {"name": "H-SEP", "zh": "美联储当时的预测（SEP 中位数，由代码换成情景表，不用大模型）", "kind": "sep"},
    "HS0": {"name": "H-SEP+FC0", "zh": "SEP 加“没有信用冲击”（金融条件中性）", "kind": "sep"},
}
TRANSLATED_ARMS = ("N1", "N2", "N3", "NW", "NS")
ORACLE_ARMS = ("OA", "OAp", "OB", "OF")
DEFAULT_RUN_ARMS = ("N1", "N2", "N3", "NW", "NS")     # what the estimate's --arms default asks the LLM for
ENGINES = {"Q": "季度模型（试运行）", "E": "月度引擎（因子轮 F）"}
CMA_ARMS = ("historical", "no_rerate")              # §4.4: both are reported; "historical" is the headline (TN record)
CMA_ZH = {"historical": "训练期历史漂移", "no_rerate": "扣除历史估值抬升"}

# ---------------------------------------------------------------- scoring (§8.6)
HORIZONS = ("1Q", "2Q", "4Q", "8Q", "end")
ASSETS = list(cq.ASSETS)
EQUITY9 = list(cq.EQUITY_LIKE)
ASSET_SETS = {"equity9": EQUITY9, "gold": ["gold"], "crude": ["crude_oil"], "rates": ["treasury_10y", "cash"],
              "us_equity": ["us_equity"], "score10": list(cq.SCORE_ASSETS), "all13": ASSETS}
ASSET_SET_ZH = {"equity9": "股票类 9 个（美股、纳指、六个行业、REITs）", "gold": "黄金", "crude": "原油",
                "rates": "国债与现金", "us_equity": "美股", "score10": "10 个风险资产", "all13": "全部 13 个资产"}
SMALL_MULTIPLE_ASSETS = ("us_equity", "nasdaq", "gold", "treasury_10y")
BOOT_DRAWS = 2000
BOOT_SEED = 20261102
BUST_DD = -0.20                                       # §8.2: busts are pre-registered as DD <= -0.20
Q_PATHS = 1000
E_PATHS = 400
SEED = 11                                             # the research scorer's seed; each episode adds its index

# ---------------------------------------------------------------- calls and confirmation (§8.5)
ESTIMATE_FRESH_S = 600.0                              # /run needs an estimate issued in the last 10 minutes
PROBE_SCHEMA_VERSION = "narrative-probe-1"
TRANSLATION_OUTPUT_TOKENS = (3000, 8000)             # §8.5 assumptions until the first run measures them
PROBE_OUTPUT_TOKENS = (100, 400)
CALL_SECONDS = (20, 60)
CONCURRENCY = 2

# ---------------------------------------------------------------- labels (Chinese, shown by the evaluation tab)
LABELS = {
    "research": "研究轮（代译，不盲，2026-11-01）",
    "legacy": "研究轮代译（无引文）",
    "recognised": "识别出年代的回放（不是盲测）",
    "mock": "模拟后端（不调用模型，不消耗订阅额度）：这是流程检验，不是叙事检验的结果",
    "trial": cq.TRIAL_LABEL,
    "identity": "恒等式上界（不是模型能力）",
    "anecdote": "每个十年只有 2–3 个独立窗口，只能当个案",
    "no_v2": "v2 叙事（盲测规范 v2）还没有写：需要一轮写作",
    "short": "训练期短的片段（单独报告，不进标题）",
}
