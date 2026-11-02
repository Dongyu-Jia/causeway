"""Distribution models of the estimated tickers (instr-mapped §5; defect 4): they split the total return into cash
and price and never change the total (T6).

Sticky payouts with error correction toward the ticker's normal yield, per quarter (annual parameters / 4):

    log D_q = log D_q-1 + [c(q) + gamma (log(4 D_q-1 / Pbar_q-1) - m)] / 4 + b_s (s_q - s_q-1) / 100
    c(q)    = c_LR + (c_SR - c_LR) exp(-q / (4 tau))           (tau = 3 years)

  Pbar   the average of the last 4 quarter-end prices (the policy reacts to a smoothed price)
  m      the ticker's average log yield in its sample (the normal yield is exp(m))
  c_SR   the last five years' payout growth; c_LR the long-run growth that keeps the yield stationary
  b_s    BDCs: + 3% of payout per +1pp of the 3-month bill (a calibrated assumption); MLPs freeze growth for 4
         quarters after a trailing-year log total return below -30% (an assumption)

c_LR from the running core (defect 4): c_LR = ppy x mean(systematic return) over the row's scenario (its group's
paths and periods) + the row's own alpha (its alpha draw, 4 x alpha_q per year) - the normal yield, i.e. the row's
expected log price growth on THIS core, so the yield stays at its normal level for decades whatever the core (the
frozen c_lr of recipes.json was calibrated on a stand-in core: SCHD's 8.3%/yr against about 4.4%/yr of price growth
in the engine's scenarios). On a path with no shocks the yield returns to normal +-0.5pp within 40 years (the test).

The explicit cut (review finding: the error correction alone is unstable at high yields). Write y = 4D/Pbar. With
c = c_LR the yield moves by about (y - y_n) + gamma (log y - m) a year before shocks: the payout drains the price
faster than gamma pulls the payout back once y is above a threshold (1.36 x normal for MPLX, gamma -0.10 and an
8.5% normal yield; 2.2 x for EPD, 2.7 x for MO), and the 2% price floor then made the drained state absorbing (a
holding paying out its whole value every quarter while its price went to 0). So when the rule's payout would put
4D/Pbar above cut_multiple x the normal yield (estimated.json payout_cut, default 2), the distribution is cut back
to the normal yield on Pbar: D = y_n Pbar / 4, an explicit cut, recorded in "cut_q". The yield then stays inside
(0, cut_multiple x y_n] on the smoothed price, and a payout the period's total return cannot carry (the 2% floor)
is a last-resort guard that the cut keeps from binding but in extreme quarters.

The other rules: XLG pays k x the core's mapped index income (0.646 x (0.71 us_equity + 0.29 nasdaq)); BRK.B
pays nothing. A payout the total return cannot carry is capped (the price keeps at least 2% of its value).
"""
from __future__ import annotations

import numpy as np

FLOOR = 0.02               # price growth factor floor of one period: a payout above it is capped


def c_lr_running(x: np.ndarray, alpha_q, normal_yield: float, ppy: int, groups: np.ndarray) -> np.ndarray:
    """(n,) c_LR per row: ppy x the mean over its group's rows and periods of the systematic return per period,
    + 4 x the row's quarterly alpha (a scalar or (n,): each row's own alpha draw), - the normal yield."""
    n = x.shape[0]
    a = np.broadcast_to(np.asarray(alpha_q, dtype=float), (n,))
    out = np.empty(n)
    for g in np.unique(groups):
        m = groups == g
        out[m] = ppy * float(x[m].mean()) + 4.0 * a[m] - normal_yield
    return out


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a / b where b > 0, else 0 (a holding whose price is 0 pays nothing per unit of its value)."""
    out = np.zeros(np.broadcast(a, b).shape)
    np.divide(a, b, out=out, where=b > 0)
    return out


def sticky_quarterly(lr_q: np.ndarray, dist: dict, state: dict, tb3_q: np.ndarray, c_lr: np.ndarray,
                     price0: float | None = None, monthly_lr: np.ndarray | None = None,
                     pay_per_year: int = 4) -> dict:
    """The sticky rule on quarterly log total returns lr_q (n, Q).

    Without monthly_lr: quarter-end prices P_q = P_q-1 (e^lr_q - D_q / P_q-1). With monthly_lr (n, Q, 3) the
    quarter's cash D_q is paid in its months (monthly payers a third a month, quarterly payers in the third
    month) and the price follows the months: P_m = P_m-1 (e^lr_m - cash_m / P_m-1); D_q itself is set from the
    quarter-end prices exactly as in the quarterly rule (it is known before the quarter but for tb3's change).
    dist["cut_multiple"] (None: no cut) caps 4 D_q / Pbar at that multiple of the normal yield by cutting D_q back
    to the normal yield on Pbar (module docstring).
    Returns {"dps_q" (n, Q), "price_q" (n, Q + 1), "cut_q" (n, Q) bool: an explicit cut in that quarter,
    "floor_q" (n, Q) bool: the 2% floor capped the payout, "cash_m" / "price_m" / "income_m" (with monthly_lr)}."""
    n, Q = lr_q.shape
    P0 = float(price0 if price0 is not None else state["price"])
    D0 = state.get("D_last_q") or state.get("yield_ttm", 0.0) * P0 / 4.0
    price = np.empty((n, Q + 1))
    price[:, 0] = P0
    D = np.full(n, float(D0))
    dps = np.zeros((n, Q))
    cut = np.zeros((n, Q), dtype=bool)
    floor = np.zeros((n, Q), dtype=bool)
    freeze = np.zeros(n, dtype=int)
    na = int(dist.get("anchor_quarters", 1))
    k_cut = dist.get("cut_multiple")
    y_n = float(dist.get("normal_yield", np.exp(dist["m"])))
    if monthly_lr is not None:
        cash_m = np.zeros((n, Q, 3))
        price_m = np.empty((n, Q * 3 + 1))
        price_m[:, 0] = P0
        share = np.full(3, 1.0 / 3.0) if pay_per_year == 12 else np.array([0.0, 0.0, 1.0])
    for q in range(Q):
        c = c_lr + (dist["c_sr"] - c_lr) * np.exp(-q / (4.0 * dist["tau_years"]))
        anchor = price[:, max(0, q + 1 - na):q + 1].mean(axis=1)
        logy = np.log(np.maximum(safe_div(4.0 * D, anchor), 1e-6))
        g = (c + dist["gamma"] * (logy - dist["m"])) / 4.0
        if dist.get("b_s"):
            s_prev = tb3_q[:, q - 1] if q > 0 else tb3_q[:, 0]
            g = g + dist["b_s"] * (tb3_q[:, q] - s_prev) / 100.0
        if dist.get("freeze_below") is not None:
            trail = lr_q[:, max(0, q - 4):q].sum(axis=1) if q > 0 else np.zeros(n)
            freeze = np.where(trail < dist["freeze_below"], 4, np.maximum(freeze - 1, 0))
            g = np.where(freeze > 0, np.minimum(g, 0.0), g)
        Dq = D * np.exp(g)
        if k_cut is not None:
            over = 4.0 * Dq > float(k_cut) * y_n * anchor
            Dq = np.where(over, y_n * anchor / 4.0, Dq)
            cut[:, q] = over
        P_start = price[:, q]
        if monthly_lr is None:
            gross = np.exp(lr_q[:, q])
            yq = safe_div(Dq, P_start)
            pg = gross - yq
            bad = pg < FLOOR
            pg = np.where(bad, FLOOR, pg)
            Dq = np.where(bad, (gross - FLOOR) * P_start, Dq)
            floor[:, q] = bad
            price[:, q + 1] = P_start * pg
        else:
            Pm = P_start.copy()
            paid = np.zeros(n)
            hit = np.zeros(n, dtype=bool)
            for j in range(3):
                gross = np.exp(monthly_lr[:, q, j])
                cm = Dq * share[j]
                pg = gross - safe_div(cm, Pm)
                bad = pg < FLOOR
                pg = np.where(bad, FLOOR, pg)
                cm = np.where(bad, (gross - FLOOR) * Pm, cm)
                hit |= bad
                cash_m[:, q, j] = cm
                paid = paid + cm
                Pm = Pm * pg
                price_m[:, 3 * q + j + 1] = Pm
            Dq = paid
            floor[:, q] = hit
            price[:, q + 1] = Pm
        dps[:, q] = Dq
        D = np.where(Dq > 0, Dq, D)
    out = {"dps_q": dps, "price_q": price, "cut_q": cut, "floor_q": floor}
    if monthly_lr is not None:
        cm = cash_m.reshape(n, Q * 3)
        out.update(cash_m=cm, price_m=price_m, income_m=safe_div(cm, price_m[:, :-1]))
    return out


def price_prop(lr_q: np.ndarray, yq: np.ndarray, price0: float, monthly_lr: np.ndarray | None = None) -> dict:
    """XLG: the quarter's cash = yq x the quarter-start price (yq from the core's index income), paid at the
    quarter's end; same outputs as sticky_quarterly."""
    n, Q = lr_q.shape
    price = np.empty((n, Q + 1))
    price[:, 0] = price0
    dps = np.zeros((n, Q))
    if monthly_lr is not None:
        cash_m = np.zeros((n, Q, 3))
        price_m = np.empty((n, Q * 3 + 1))
        price_m[:, 0] = price0
    for q in range(Q):
        P_start = price[:, q]
        Dq = yq[:, q] * P_start
        if monthly_lr is None:
            gross = np.exp(lr_q[:, q])
            pg = np.maximum(gross - safe_div(Dq, P_start), FLOOR)
            Dq = (gross - pg) * P_start
            price[:, q + 1] = P_start * pg
        else:
            Pm = P_start.copy()
            for j in range(3):
                gross = np.exp(monthly_lr[:, q, j])
                cm = Dq if j == 2 else np.zeros(n)
                pg = np.maximum(gross - safe_div(cm, Pm), FLOOR)
                cm = (gross - pg) * Pm if j == 2 else cm
                cash_m[:, q, j] = cm
                Pm = Pm * pg
                price_m[:, 3 * q + j + 1] = Pm
            Dq = cash_m[:, q].sum(axis=1)
            price[:, q + 1] = Pm
        dps[:, q] = Dq
    out = {"dps_q": dps, "price_q": price}
    if monthly_lr is not None:
        cm = cash_m.reshape(n, Q * 3)
        out.update(cash_m=cm, price_m=price_m, income_m=safe_div(cm, price_m[:, :-1]))
    return out


def no_payout(lr_q: np.ndarray, price0: float, monthly_lr: np.ndarray | None = None) -> dict:
    n, Q = lr_q.shape
    price = price0 * np.concatenate([np.ones((n, 1)), np.exp(np.cumsum(lr_q, axis=1))], axis=1)
    out = {"dps_q": np.zeros((n, Q)), "price_q": price}
    if monthly_lr is not None:
        lm = monthly_lr.reshape(n, Q * 3)
        pm = price0 * np.concatenate([np.ones((n, 1)), np.exp(np.cumsum(lm, axis=1))], axis=1)
        out.update(cash_m=np.zeros((n, Q * 3)), price_m=pm, income_m=np.zeros((n, Q * 3)))
    return out
