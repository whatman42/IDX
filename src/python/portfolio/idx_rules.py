"""IDX microstructure rules for paper simulation (explicit assumptions)."""

from __future__ import annotations

import math


def tick_size(price: float) -> float:
    if price < 200:
        return 1.0
    if price < 500:
        return 2.0
    if price < 2000:
        return 5.0
    if price < 5000:
        return 10.0
    return 25.0


def round_to_tick(price: float, side: int) -> float:
    t = tick_size(price)
    if t <= 0:
        return price
    if side >= 0:
        return math.ceil(price / t) * t
    return math.floor(price / t) * t


BOARD_LOT = 100


def round_lot(qty: float, lot: float = BOARD_LOT) -> float:
    return float(int(qty / lot) * lot)
