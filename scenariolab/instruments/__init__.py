"""Instrument layer (标的层): 16 tickers simulated as recipes on the core simulation's own paths.

Every ticker is a recipe applied to a core view (CoreView: the monthly engine, the quarterly model being designed,
or the agent tree's particles) and returns per-period total returns, cash income and USD price levels in the
shape of the 13 asset classes, so portfolios and withdrawals take tickers as extra columns.

  structural  QQQ, QLD, TQQQ, QQQI, TLT, GLD (and GOLD, the core's gold asset itself): derived from the
              products' mechanics -- daily-reset leverage with the period's variance, written NDX calls priced
              at 0.855 x VXN, the long-yield model with duration and convexity, fund fees
  estimated   XLG, SCHD, MO, BRK.B, MAIN, ARCC, CSWC, EPD, MPLX, BME: quarterly style portfolios of the core's
              asset classes + factors + the ticker's own residual, estimated on 13F quarter-end prices from 2013

Modules: core (CoreView, bridges), vol (the joint volatility state), structural, estimated, distributions,
layer (run_layer, InstrumentSet), fees (the one fee table), data / import_raw / refresh (the data layer),
fit (explicit refits), validate (historical replays), build (the build step), catalog (UI metadata).
"""
from __future__ import annotations

STRUCTURAL = ("QQQ", "QLD", "TQQQ", "QQQI", "TLT", "GLD")
ESTIMATED = ("XLG", "SCHD", "MO", "BRK.B", "MAIN", "ARCC", "CSWC", "EPD", "MPLX", "BME")
TICKERS = STRUCTURAL + ESTIMATED          # the 16 stored columns
ALIASES = {"GOLD": "gold"}                # GOLD (金价) is the core's gold asset itself: no column of its own
LEVERAGED = ("QLD", "TQQQ")
