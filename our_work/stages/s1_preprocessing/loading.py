"""S1a - load and clean the raw WUSTL-EHMS-2020 CSV (identical to notebook Stage 1/2)."""
from pathlib import Path

import numpy as np
import pandas as pd

OUR_WORK = Path(__file__).resolve().parents[2]
DEFAULT_CSV = OUR_WORK / "data" / "wustl-ehms-2020_with_attacks_categories (1).csv"
DROP_COLS = ["Dir", "Flgs", "SrcAddr", "DstAddr", "SrcMac", "DstMac", "Packet_num", "Attack Category"]
N_FEATURES = 36


def load_clean(csv=None):
    """dropna -> drop_duplicates -> drop 8 non-predictive columns -> coerce `Sport` to numeric.

    Returns (X float64 array with 36 columns, y int64 array of 0/1)."""
    df = pd.read_csv(csv or DEFAULT_CSV)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.drop(columns=DROP_COLS)
    X = df.drop("Label", axis=1)
    y = df["Label"]
    X["Sport"] = pd.to_numeric(X["Sport"], errors="coerce")   # mixes ports and service names like 'http'
    X.dropna(inplace=True)
    y = y[X.index]
    assert X.shape[1] == N_FEATURES, X.shape
    return X.values.astype(np.float64), y.values.astype(np.int64)
