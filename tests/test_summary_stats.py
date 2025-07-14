import pandas as pd
import os
from scripts.data_load import load_crypto_data

def test_summary_statistics():
    symbol = "LTC"
    filepath = os.path.join("Dados", "Dia", f"Poloniex_{symbol}USDT_d.csv")
    df = load_crypto_data(filepath)
    stats = df["Close"].describe()
    assert stats["count"] > 1000
    assert stats["mean"] > 0
    assert stats["std"] > 0
