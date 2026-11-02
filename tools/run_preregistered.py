"""CLI of the pre-registered runs T-A, T-C and T-S: the code lives in scenariolab/evaluation/protocol.py (SIM_DESIGN
§3.9), which the opt-in build step `evaluation` also runs. Every name of that module is available here, so
`tools/run_preregistered.py` stays the documented entry point (see the module's docstring for the commands).

    python tools/run_preregistered.py check [--inputs]
    python tools/run_preregistered.py freeze --only TC --tag r2 --supersedes 2026-11-02_TC.json
    python tools/run_preregistered.py run TC --tag r2
    python tools/run_preregistered.py note TA --text "..."
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenariolab.evaluation import protocol as _protocol  # noqa: E402

globals().update({k: v for k, v in vars(_protocol).items() if not k.startswith("__")})

if __name__ == "__main__":
    sys.exit(_protocol.main())
