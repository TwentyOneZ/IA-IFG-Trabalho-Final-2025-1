import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.data_load import load_crypto_data
from scripts.features import create_features
from scripts.models import get_models
from scripts.utils import simular_lucro_vetorizado
from sklearn.preprocessing import PolynomialFeatures

def main():
    parser = argparse.ArgumentParser(description="Simula lucro com base nas previsões de modelo")
    parser.add_argument("--crypto", required=True, help="Símbolo da criptomoeda (ex: BTC)")
    parser.add_argument("--model", required=True, help="Modelo a usar: linear, mlp, poly_deg2, etc.")
    parser.add_argument("--start-date", type=str, help="Data inicial da simulação (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true", help="Salvar gráfico e planilha em figures/lucro/")
    args = parser.parse_args()

    symbol = args.crypto.upper()
    model_name = args.model
    filepath = os.path.join("Dados", "Dia", f"Poloniex_{symbol}USDT_d.csv")

    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return

    df = load_crypto_data(filepath)
    df = create_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if args.start_date:
        try:
            df = df[df["Date"] >= pd.to_datetime(args.start_date)]
        except Exception as e:
            print(f"Data inválida: {args.start_date}. Use o formato YYYY-MM-DD.")
            return

    X = df[["pct_change_1d", "volume_change_1d", "ma_3", "ma_7", "ma_14", "std_7", "std_14"]].values
    y = df["target"].values

    models = get_models(degree_list=[2, 3, 5])
    if model_name not in models:
        print(f"Modelo '{model_name}' não encontrado.")
        return

    model = models[model_name]
    if isinstance(model, tuple) and model[0] == "poly":
        poly = model[1]
        X = poly.fit_transform(X)
        model = model[2]

    model.fit(X, y)
    df["target"] = model.predict(X)

    # Simulação com modelo e buy & hold
    resultado_modelo = simular_lucro_vetorizado(df)

    # Criação da pasta
    os.makedirs("figures/lucro", exist_ok=True)

    # Gráfico
    plt.figure(figsize=(12, 5))
    plt.plot(resultado_modelo["Data"], resultado_modelo["capital_estrategia"], label="Modelo (estratégia ativa)")
    plt.plot(resultado_modelo["Data"], resultado_modelo["capital_hold"], label="Buy and Hold", linestyle="--")
    plt.title(f"Simulação de Lucro — {symbol} com {model_name}")
    plt.xlabel("Data")
    plt.ylabel("Capital acumulado (USD)")
    plt.grid(True)
    plt.legend()

    # Nomes personalizados
    sufixo_data = f"_desde_{args.start_date}" if args.start_date else ""
    base_nome = f"lucro_{symbol}_{model_name}{sufixo_data}"

    # Exporta gráfico
    if args.save:
        img_path = f"figures/lucro/{base_nome}.png"
        plt.savefig(img_path, dpi=150)
        print(f"📈 Gráfico salvo em {img_path}")
    else:
        plt.show()

    # Exporta planilha
    if args.save:
        csv_path = f"figures/lucro/{base_nome}.csv"
        resultado_modelo.to_csv(csv_path, index=False)
        print(f"📄 Planilha salva em {csv_path}")

if __name__ == "__main__":
    main()
