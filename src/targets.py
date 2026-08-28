#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:47:08 2026

@author: anasantana
"""

import numpy as np
import pandas as pd

from data import load_matrix, load_returns

DEMEAN_WINDOW = 63
EPS = 1e-3

def _log_and_ddemean(panel: pd.DataFrame) -> pd.DataFrame:
    logged = np.log(panel + EPS)
    trailing = logged.rolling(DEMEAN_WINDOW, min_periods=21).mean().shift(1)
    out = (logged - trailing).dropna(how="all")
    return out

def _universe_columns(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[load_returns("top100").columns]

def abs_return_panel() -> pd.DataFrame():
    r = np.log1p(load_returns("top100"))
    return _log_and_ddemean(r.abs())

def volume_panel() -> pd.DataFrame():
    vol = _universe_columns(load_matrix("volMM"))
    return _log_and_ddemean(vol)

def rv_panel(window: int = 21) -> pd.DataFrame:
    r = np.log1p(load_returns("top100"))
    rv = (r**2).rolling(window).mean().pow(0.50)
    return  _log_and_ddemean(rv)

TARGETS = {
    "returns": None,
    "absret": abs_return_panel,
    "volume": volume_panel,
    "rv": rv_panel,
}

if __name__ == "__main__":
    for name, fn in TARGETS.items():
        if fn is None:
            continue
        p = fn()
        ac = p.apply(lambda s: s.autocorr()).mean()
        print(f"{name:8s} panel {p.shape}, mean lag-1 autocorrelation {ac:+.3f}")