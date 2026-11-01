"""EngineContext: the fitted state every analysis and API call shares."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from . import config, jsonio
from .data.panel import load_meta, load_panel
from .engine.factor_model import AssetFactorModel, ModelSpec
from .engine.simulate import Anchor, MonthDB, capital_market_assumptions, make_anchor

SHOCKS_PATH = config.PROCESSED_DIR / "structural_shocks.csv"
SELECTION_PATH = config.RESULTS_DIR / "model_selection.json"


def load_panel_with_shocks() -> pd.DataFrame:
    panel = load_panel()
    if SHOCKS_PATH.exists():
        shocks = pd.read_csv(SHOCKS_PATH, index_col="date", parse_dates=["date"])
        shocks.index = shocks.index.to_period("M").to_timestamp()
        for col in shocks.columns:
            if col.startswith("e_"):
                panel[col] = shocks[col].reindex(panel.index)
    return panel


def selected_spec() -> ModelSpec | None:
    if SELECTION_PATH.exists():
        try:
            return ModelSpec.from_dict(jsonio.load(SELECTION_PATH)["spec"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


@dataclass
class EngineContext:
    panel: pd.DataFrame
    meta: dict
    spec: ModelSpec
    model: AssetFactorModel
    db: MonthDB
    anchor: Anchor
    cma: dict

    @classmethod
    def load(cls, selection: bool = True, spec: ModelSpec | None = None,
             cma_method: str = "valuation_adjusted") -> "EngineContext":
        panel = load_panel_with_shocks()
        meta = load_meta()
        if spec is None and selection:
            spec = selected_spec()
        spec = spec or ModelSpec()
        missing = [c for c in spec.shock_cols if c not in panel.columns]
        if missing:  # selection refers to shocks that are not built yet
            spec = ModelSpec.from_dict({**spec.to_dict(), "shock_cols": []})
        model = AssetFactorModel(spec).fit(panel, config.MODEL_START, None)
        db = MonthDB(panel, model)
        anchor = make_anchor(panel, model)
        cma = capital_market_assumptions(db, anchor, method=cma_method)
        return cls(panel=panel, meta=meta, spec=spec, model=model, db=db, anchor=anchor, cma=cma)

    def fit(self, spec: ModelSpec, start=config.MODEL_START, end=None) -> AssetFactorModel:
        return AssetFactorModel(spec).fit(self.panel, start, end)


@lru_cache(maxsize=1)
def get_context() -> EngineContext:
    """Process-wide context for the web server (call reset_context() after a build)."""
    return EngineContext.load(selection=True)


def reset_context() -> None:
    get_context.cache_clear()
