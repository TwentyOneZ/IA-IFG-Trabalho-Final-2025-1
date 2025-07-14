import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from scripts.data_load import load_crypto_data
from scripts.features import create_features
from scripts.models import get_models
from scripts.utils import simular_lucro_vetorizado

# Configuração do logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def residual_standard_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Erro Padrão Residual (RSE) entre os valores reais e previstos.

    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores previstos.

    Returns:
        float: Erro padrão residual.
    """
    n = len(y_true)
    sse = np.sum((y_true - y_pred) ** 2)
    return np.sqrt(sse / (n - 2))

def main() -> None:
    """
    Executa o pipeline completo de comparação entre modelos preditivos
    para uma criptomoeda selecionada, salvando gráficos e métricas em arquivos.
    """
    parser = argparse.ArgumentParser(description="Compara modelos preditivos para criptomoeda")
    parser.add_argument("--crypto", required=True, help="Símbolo da criptomoeda (ex: BTC)")
    args = parser.parse_args()

    symbol = args.crypto.upper()
    filepath = f"Dados/Dia/Poloniex_{symbol}USDT_d.csv"
    if not os.path.exists(filepath):
        logging.error(f"Arquivo não encontrado: {filepath}")
        return

    logging.info("Carregando e preparando os dados...")

    # Carrega e prepara os dados
    df = load_crypto_data(filepath)
    df = create_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.sort_values("Date").reset_index(drop=True)

    # split treino/teste por data
    split_date = pd.to_datetime("2024-01-01")
    train_df = df[df["Date"] < split_date].copy()
    test_df  = df[df["Date"] >= split_date].copy()

    feature_cols = ["pct_change_1d","volume_change_1d","ma_3","ma_7","ma_14","std_7","std_14"]
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["target"].values

    # padroniza based on treino
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = get_models(degree_list=list(range(2, 11)))
    results = {}

    # calcula métricas e armazena detalhes
    logging.info("Treinando modelos e calculando métricas...")
    for name, m in models.items():
        if isinstance(m, tuple) and m[0] == "poly":
            _, poly, reg = m
            model = reg
            X_tr = poly.fit_transform(X_train_scaled)
            X_te = poly.transform(X_test_scaled)
            degree = poly.degree
        else:
            model = m
            X_tr, X_te = X_train_scaled, X_test_scaled
            degree = None

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        corr, _ = pearsonr(y_test, y_pred)
        rse = residual_standard_error(y_test, y_pred)

        coef = getattr(model, "coef_", None)
        intercept = getattr(model, "intercept_", None)
        if coef is not None:
            terms = [f"{intercept:.2f}"]
            for i, c in enumerate(coef):
                if abs(c) >= 0.0001:
                    terms.append(f"{c:.4f}·x{i}")
            eq_full = " + ".join(terms)
        else:
            eq_full = "n/a"

        if degree:
            eq_summary = f"polynomial deg {degree} ({len(terms)-1} terms)"
        else:
            eq_summary = "linear"

        results[name] = {
            "corr": corr,
            "rse": rse,
            "eq_summary": eq_summary,
            "eq_full": eq_full
        }

    # e) Identifica o melhor regressor pela maior correlação
    best = max(results, key=lambda k: results[k]["corr"])
    rse_mlp  = results["mlp"]["rse"]
    rse_best = results[best]["rse"]
    diff_rse = abs(rse_mlp - rse_best)

    # resumo no console
    logging.info(f"\n=== Resumo (teste) — {symbol} ===")
    for name, stats in results.items():
        logging.info(f"{name:12} | corr: {stats['corr']:.4f} | RSE: {stats['rse']:.2f} | {stats['eq_summary']}")
    logging.info(f"\nMelhor regressor: {best}")
    logging.info(f"RSE MLP: {rse_mlp:.2f} | RSE {best}: {rse_best:.2f} | ∆RSE = {diff_rse:.2f}")

    # salva detalhes completos e o ∆RSE em TXT
    os.makedirs("figures/compare", exist_ok=True)
    out_txt = f"figures/compare/results_{symbol}.txt"
    with open(out_txt, "w") as f:
        f.write(f"Detalhamento completo — {symbol}\n")
        f.write(f"Treino até {split_date.date()}, teste a partir de {split_date.date()}\n\n")
        for name, stats in results.items():
            f.write(f"Modelo: {name}\n")
            f.write(f"  Correlação: {stats['corr']:.6f}\n")
            f.write(f"  RSE       : {stats['rse']:.6f}\n")
            f.write(f"  Equação   : {stats['eq_full']}\n\n")
        f.write(f"Melhor regressor: {best}\n")
        f.write(f"RSE MLP       : {rse_mlp:.6f}\n")
        f.write(f"RSE {best}    : {rse_best:.6f}\n")
        f.write(f"Delta RSE     : {diff_rse:.6f}\n")
    logging.info(f"Detalhes completos salvos em {out_txt}")

    plt.figure(figsize=(12, 8))
    for name, m in models.items():
        # 1) monta X_tr, X_te e model exatamente como no cálculo de métricas
        if isinstance(m, tuple) and m[0] == "poly":
            _, poly, reg = m
            model = reg
            X_tr = poly.fit_transform(X_train_scaled)
            X_te = poly.transform(X_test_scaled)
        else:
            model = m
            X_tr, X_te = X_train_scaled, X_test_scaled

        # 2) treina no treino e prediz no teste
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        # 3) plota cada modelo com seu label
        plt.scatter(y_test, y_pred, s=5, alpha=0.5, label=name)

    # reta identidade e limite de eixo
    mn, mx = y_test.min(), y_test.max()
    plt.plot([mn, mx], [mn, mx], "k--")
    ymax = 2 * mx
    plt.ylim(0, ymax)

    plt.xlabel("Real (target)")
    plt.ylabel("Previsto")
    plt.title(f"Scatter Real vs Previsto (teste) — {symbol}")
    plt.legend(markerscale=3)
    plt.savefig(f"figures/compare/scatter_test_{symbol}.png", dpi=150)
    plt.close()

    investimento = 1000.0

    # --- evolução do capital sobre teste ---
    plt.figure(figsize=(12, 6))
    for name, m in models.items():
        # prepara e treina igual acima
        if isinstance(m, tuple) and m[0] == "poly":
            _, poly, reg = m
            model = reg
            X_te = poly.transform(X_test_scaled)
            X_tr = poly.fit_transform(X_train_scaled)
        else:
            model = m
            X_te = X_test_scaled
            X_tr = X_train_scaled

        model.fit(X_tr, y_train)

        # simula capital: copia somente o test_df
        tmp = test_df.copy().reset_index(drop=True)
        tmp["target"] = model.predict(X_te)
        sim = simular_lucro_vetorizado(tmp, investimento_inicial=investimento)
        plt.plot(sim["Data"], sim["capital_estrategia"], label=name)

    # buy & hold sobre teste
    hold = test_df[["Date", "Close"]].copy()
    hold["daily_return"] = hold["Close"] / hold["Close"].shift(1)
    hold.loc[hold.index[0], "daily_return"] = 1.0
    hold["capital_hold"] = investimento * hold["daily_return"].cumprod()
    plt.plot(hold["Date"], hold["capital_hold"], "k--", label="hold")

    plt.title(f"Evolução do capital (teste) — {symbol}")
    plt.xlabel("Data")
    plt.ylabel("Capital (USD)")
    plt.legend()
    plt.grid()
    plt.savefig(f"figures/compare/profit_test_{symbol}.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
