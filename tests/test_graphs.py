import matplotlib
matplotlib.use("Agg")

import os
from scripts.data_load import load_crypto_data
from main import gerar_graficos

def test_graph_generation_creates_files():
    symbol = "LTC"
    filepath = os.path.join("Dados", "Dia", f"Poloniex_{symbol}USDT_d.csv")
    df = load_crypto_data(filepath)
    gerar_graficos(df, symbol, show=False, save=True)

    assert os.path.exists(f"figures/boxplot/boxplot_{symbol}.png")
    assert os.path.exists(f"figures/hist/hist_{symbol}.png")
    assert os.path.exists(f"figures/lineplot/lineplot_{symbol}.png")
