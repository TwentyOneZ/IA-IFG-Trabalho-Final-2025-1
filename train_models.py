import argparse
import os
import pandas as pd
import numpy as np
from scripts.data_load import load_crypto_data
from scripts.features import create_features
from scripts.models import get_models, train_model_cv
from sklearn.preprocessing import PolynomialFeatures

def main():
    parser = argparse.ArgumentParser(description="Treina modelos preditivos para fechamento de criptomoedas")
    parser.add_argument("--crypto", required=True, help="Símbolo da criptomoeda (ex: BTC)")
    parser.add_argument("--kfolds", type=int, default=5, help="Número de divisões do K-Fold (default=5)")
    args = parser.parse_args()

    symbol = args.crypto.upper()
    filepath = os.path.join("Dados", "Dia", f"Poloniex_{symbol}USDT_d.csv")
    
    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return

    print(f"\n📊 Carregando dados de {symbol}...")
    df = load_crypto_data(filepath)
    df = create_features(df)

    # Limpeza extra
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    feature_cols = [
        "pct_change_1d", "volume_change_1d",
        "ma_3", "ma_7", "ma_14",
        "std_7", "std_14"
    ]
    target_col = "target"

    X = df[feature_cols].values
    y = df[target_col].values

    models = get_models(degree_list=[2, 3, 5])

    print(f"\n🚀 Treinando modelos com validação K-Fold (k={args.kfolds})...")
    for name, m in models.items():
        if isinstance(m, tuple) and m[0] == "poly":
            poly = m[1]
            X_poly = poly.fit_transform(X)
            result = train_model_cv(X_poly, y, m[2], k=args.kfolds)
        else:
            result = train_model_cv(X, y, m, k=args.kfolds)

        print(f"\n📌 Modelo: {name}")
        print(f"RMSE médio: {result['rmse_mean']:.2f} ± {result['rmse_std']:.2f}")
        print(f"MAE  médio: {result['mae_mean']:.2f} ± {result['mae_std']:.2f}")

if __name__ == "__main__":
    main()
