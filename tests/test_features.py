import os
import pandas as pd
from scripts.data_load import load_crypto_data
from scripts.features import create_features

def test_feature_engineering_creates_expected_columns():
    filepath = os.path.join("Dados", "Dia", "Poloniex_BTCUSDT_d.csv")
    df = load_crypto_data(filepath)
    df_feat = create_features(df)

    expected_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "pct_change_1d", "volume_change_1d",
        "ma_3", "ma_7", "ma_14",
        "std_7", "std_14",
        "target"
    ]

    for col in expected_cols:
        assert col in df_feat.columns, f"Coluna esperada nao encontrada: {col}"

def test_feature_output_not_empty_or_nan():
    filepath = os.path.join("Dados", "Dia", "Poloniex_ETHUSDT_d.csv")
    df = load_crypto_data(filepath)
    df_feat = create_features(df)

    assert not df_feat.empty, "O DataFrame final esta vazio"
    assert df_feat.isna().sum().sum() == 0, "Ainda existem valores NaN no DataFrame resultante"

def test_target_column_is_shifted():
    filepath = os.path.join("Dados", "Dia", "Poloniex_LTCUSDT_d.csv")
    df = load_crypto_data(filepath)
    df_feat = create_features(df)

    # Verifica se o target e igual ao valor de fechamento do proximo dia
    assert abs(df_feat["target"].iloc[0] - df_feat["Close"].iloc[1]) < 1e-6
