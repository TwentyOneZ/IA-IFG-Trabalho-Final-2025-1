import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    para uma criptomoeda selecionada, usando uma data de corte para treino/teste,
    salvando gráficos e métricas em arquivos.
    """
    parser = argparse.ArgumentParser(description="Compara modelos preditivos para criptomoeda")
    parser.add_argument("--crypto", required=True,
                        help="Símbolo da criptomoeda (ex: BTC)")
    parser.add_argument("--start-date", type=str, default="2024-01-01",
                        help="Data de início dos dados de teste (YYYY-MM-DD). "
                             "Treino usará dados anteriores a esta data.")
    args = parser.parse_args()

    symbol = args.crypto.upper()
    filepath = f"Dados/Dia/Poloniex_{symbol}USDT_d.csv"
    if not os.path.exists(filepath):
        logging.error(f"Arquivo não encontrado: {filepath}")
        return

    logging.info("Carregando e preparando os dados...")
    df = load_crypto_data(filepath)
    df = create_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.sort_values("Date").reset_index(drop=True)

    # converte start-date e faz split treino/teste
    try:
        split_date = pd.to_datetime(args.start_date)
    except Exception:
        logging.error(f"Data inválida: {args.start_date}. Use formato YYYY-MM-DD.")
        return

    train_df = df[df["Date"] < split_date].copy()
    test_df  = df[df["Date"] >= split_date].copy()
    if train_df.empty or test_df.empty:
        logging.error("Split resultou em conjunto de treino ou teste vazio. "
                      "Verifique a --start-date fornecida.")
        return

    feature_cols = ["pct_change_1d","volume_change_1d","ma_3","ma_7","ma_14","std_7","std_14"]
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["target"].values

    # padronização com base no treino
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = get_models(degree_list=list(range(2, 11)))
    results = {}

    logging.info("Treinando modelos e calculando métricas sobre dados de teste...")
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

        eq_summary = (f"polynomial deg {degree} ({len(terms)-1} terms)"
                      if degree else "linear")

        results[name] = {
            "corr": corr,
            "rse": rse,
            "eq_summary": eq_summary,
            "eq_full": eq_full
        }

    # identifica o melhor regressor pelo maior rse
    best = max(results, key=lambda k: results[k]["corr"])
    rse_mlp  = results["mlp"]["rse"]
    rse_best = results[best]["rse"]
    diff_rse = abs(rse_mlp - rse_best)

    # resumo no console
    logging.info(f"\n=== Resumo (teste) — {symbol} — teste desde {split_date.date()} ===")
    for name, stats in results.items():
        logging.info(f"{name:12} | corr: {stats['corr']:.4f} | "
                     f"RSE: {stats['rse']:.2f} | {stats['eq_summary']}")
    logging.info(f"\nMelhor regressor: {best}")
    logging.info(f"RSE MLP: {rse_mlp:.2f} | RSE {best}: {rse_best:.2f} | ΔRSE = {diff_rse:.2f}")

    # salva detalhes completos
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

    # scatter plot
    plt.figure(figsize=(12, 8))
    for name, m in models.items():
        if isinstance(m, tuple) and m[0] == "poly":
            _, poly, reg = m
            model = reg
            X_te = poly.transform(X_test_scaled)
            X_tr = poly.fit_transform(X_train_scaled)
        else:
            model = m
            X_te, X_tr = X_test_scaled, X_train_scaled

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        plt.scatter(y_test, y_pred, s=5, alpha=0.5, label=name)

    mn, mx = y_test.min(), y_test.max()
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.ylim(0, 2 * mx)
    plt.xlabel("Real (target)")
    plt.ylabel("Previsto")
    plt.title(f"Scatter Real vs Previsto (teste) — {symbol}")
    plt.legend(markerscale=3)
    plt.savefig(f"figures/compare/scatter_test_{symbol}.png", dpi=150)
    plt.close()

    # evolução do capital sobre teste
    investimento = 1000.0
    plt.figure(figsize=(12, 6))
    for name, m in models.items():
        if isinstance(m, tuple) and m[0] == "poly":
            _, poly, reg = m
            model = reg
            X_te = poly.transform(X_test_scaled)
            X_tr = poly.fit_transform(X_train_scaled)
        else:
            model = m
            X_te, X_tr = X_test_scaled, X_train_scaled

        model.fit(X_tr, y_train)
        tmp = test_df.copy().reset_index(drop=True)
        tmp["target"] = model.predict(X_te)
        sim = simular_lucro_vetorizado(tmp, investimento_inicial=investimento)
        plt.plot(sim["Data"], sim["capital_estrategia"], label=name)

    hold = test_df[["Date","Close"]].copy()
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
