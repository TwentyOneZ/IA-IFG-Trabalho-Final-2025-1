import os
from scripts.data_load import load_crypto_data

def test_load_btc_data():
    symbol = "LTC"
    filepath = os.path.join("Dados", "Dia", f"Poloniex_{symbol}USDT_d.csv")
    df = load_crypto_data(filepath)
    assert df is not None, "O DataFrame retornado e None"
    assert not df.empty, "O DataFrame esta vazio"
    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"], "Colunas incorretas"
