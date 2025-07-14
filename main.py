import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
import scipy.stats as stats
import subprocess, sys

from scripts.data_load import load_crypto_data
from scripts.features import create_features
from scripts.models import get_models
from scripts.utils import simular_lucro_vetorizado
from sklearn.preprocessing import PolynomialFeatures
from scripts.models import train_model_cv
import logging

# ==== ANALYZE ====

def gerar_graficos(df: pd.DataFrame, symbol: str, show: bool, save: bool) -> None:
    """
    Gera gráficos estatísticos para uma criptomoeda: boxplot, histograma e gráfico de linha com média, mediana e moda.

    Args:
        df (pd.DataFrame): DataFrame com dados da criptomoeda.
        symbol (str): Símbolo da criptomoeda (ex: BTC).
        show (bool): Se True, exibe os gráficos.
        save (bool): Se True, salva os gráficos na pasta 'figures/'.
    """
    paths = {
        "boxplot": os.path.join("figures", "boxplot"),
        "hist": os.path.join("figures", "hist"),
        "lineplot": os.path.join("figures", "lineplot"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    # Boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(y=df["Close"])
    plt.title(f"Boxplot do Preço de Fechamento - {symbol}")
    if save:
        plt.savefig(f"{paths['boxplot']}/boxplot_{symbol}.png", dpi=150)
    if show:
        plt.show()
    plt.close()

    # Histograma
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Close"], bins=30, kde=True)
    plt.title(f"Histograma do Preço de Fechamento - {symbol}")
    if save:
        plt.savefig(f"{paths['hist']}/hist_{symbol}.png", dpi=150)
    if show:
        plt.show()
    plt.close()

    # Linha com média, mediana e moda
    df["Mean"] = df["Close"].expanding().mean()
    df["Median"] = df["Close"].expanding().median()
    df["Mode"] = df["Close"].rolling(window=10).apply(lambda x: stats.mode(x)[0], raw=True)

    plt.figure(figsize=(12, 4))
    plt.plot(df["Date"], df["Close"], label="Close", alpha=0.6)
    plt.plot(df["Date"], df["Mean"], label="Média")
    plt.plot(df["Date"], df["Median"], label="Mediana")
    plt.plot(df["Date"], df["Mode"], label="Moda", linestyle='--')
    plt.title(f"Evolução do Preço com Média, Mediana e Moda - {symbol}")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"{paths['lineplot']}/lineplot_{symbol}.png", dpi=150)
    if show:
        plt.show()
    plt.close()

def analyze(args: argparse.Namespace) -> None:
    """
    Função principal para o comando 'analyze'. Carrega os dados e gera os gráficos.

    Args:
        args (argparse.Namespace): Argumentos passados pela CLI.
    """
    symbol = args.crypto.upper()
    filepath = f"Dados/Dia/Poloniex_{symbol}USDT_d.csv"

    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return

    df = load_crypto_data(filepath)
    close = df["Close"]
    logging.info(f"Média: {close.mean():.2f}")
    logging.info(f"Mediana: {close.median():.2f}")
    logging.info(f"Moda: {close.mode().iloc[0]:.2f}")
    logging.info(f"Variância: {close.var():.2f}")
    logging.info(f"Desvio padrão: {close.std():.2f}")
    logging.info(f"1º quartil (q1): {np.percentile(close, 25):.2f}")
    logging.info(f"3º quartil (q3): {np.percentile(close, 75):.2f}")
    logging.info(f"Mínimo: {close.min():.2f}")
    logging.info(f"Máximo: {close.max():.2f}")
    gerar_graficos(df, symbol, args.show, args.save)

# ==== SIMULATE ====

def simulate(args: argparse.Namespace) -> None:
    """
    Executa a simulação de lucro utilizando um modelo preditivo para a criptomoeda especificada.

    Args:
        args (argparse.Namespace): Argumentos da linha de comando contendo crypto, model, start_date e save.
    """
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
    resultado = simular_lucro_vetorizado(df)

    os.makedirs("figures/lucro", exist_ok=True)
    sufixo_data = f"_desde_{args.start_date}" if args.start_date else ""
    base_nome = f"lucro_{symbol}_{model_name}{sufixo_data}"

    plt.figure(figsize=(12, 5))
    plt.plot(resultado["Data"], resultado["capital_estrategia"], label="Modelo (estratégia ativa)")
    plt.plot(resultado["Data"], resultado["capital_hold"], label="Buy and Hold", linestyle="--")
    plt.title(f"Simulação de Lucro — {symbol} com {model_name}")
    plt.xlabel("Data")
    plt.ylabel("Capital acumulado (USD)")
    plt.grid(True)
    plt.legend()

    if args.save:
        plt.savefig(f"figures/lucro/{base_nome}.png", dpi=150)
        print(f"📈 Gráfico salvo em figures/lucro/{base_nome}.png")
    else:
        plt.show()

    if args.save:
        resultado.to_csv(f"figures/lucro/{base_nome}.csv", index=False)
        print(f"📄 Planilha salva em figures/lucro/{base_nome}.csv")

# ==== TRAIN ====

def train(args: argparse.Namespace) -> None:
    """
    Treina múltiplos modelos preditivos usando validação cruzada (K-Fold) com os dados da criptomoeda.

    Args:
        args (argparse.Namespace): Argumentos da CLI contendo símbolo da criptomoeda e número de folds.
    """
    symbol = args.crypto.upper()
    filepath = os.path.join("Dados", "Dia", f"Poloniex_{symbol}USDT_d.csv")

    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return

    df = load_crypto_data(filepath)
    df = create_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df[["pct_change_1d", "volume_change_1d", "ma_3", "ma_7", "ma_14", "std_7", "std_14"]].values
    y = df["target"].values

    print(f"\n🚀 Treinando modelos com validação K-Fold (k={args.kfolds})...")
    models = get_models(degree_list=[2, 3, 5])

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


# ==== MAIN ====

def main() -> None:
    """
    Função principal que configura e executa a interface de linha de comando (CLI),
    redirecionando para os subcomandos apropriados.
    """
    parser = argparse.ArgumentParser(description="Painel de controle de criptomoedas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Gera gráficos (boxplot, histograma e evolução do preço com média, mediana e moda da criptomoeda")
    analyze_parser.add_argument("--crypto", required=True, help="Símbolo da criptomoeda (ex: BTC)")
    analyze_parser.add_argument("--show", action="store_true", help="Exibe os gráficos")
    analyze_parser.add_argument("--save", action="store_true", help="Salva os gráficos em 'figures/'")

    # simulate
    simulate_parser = subparsers.add_parser("simulate", help="Simula lucro com base nas previsões do modelo")
    simulate_parser.add_argument("--crypto", required=True, help="Símbolo da criptomoeda (ex: BTC)")
    simulate_parser.add_argument("--model", required=True, help="Modelo a usar: linear, mlp, poly_deg2 a poly_deg10.")
    simulate_parser.add_argument("--start-date", type=str, help="Data inicial da simulação (YYYY-MM-DD)")
    simulate_parser.add_argument("--save", action="store_true", help="Salva gráfico e planilha")

    # train
    train_parser = subparsers.add_parser("train", help="Treina e avalia modelos com K-Fold")
    train_parser.add_argument("--crypto", required=True, help="Símbolo da criptomoeda (ex: BTC)")
    train_parser.add_argument("--kfolds", type=int, default=5, help="Número de folds para validação (default=5)")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compara todos os modelos com treino até 01/01/2024 e teste a partir do dia seguinte até a última amostra. Gera gráficos/textos.")
    compare_parser.add_argument("--crypto", required=True, help="Símbolo da criptomoeda (ex: BTC)")

    # hypothesis
    hypothesis_parser = subparsers.add_parser(
        "hypothesis",
        help="Teste de hipótese de retorno médio diário ≥ x%%"
    )
    hypothesis_parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Valor x em porcentagem (ex: 1.5 para 1.5%%)"
    )
    hypothesis_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nível de significância (default 0.05)"
    )
    hypothesis_parser.add_argument(
        "--save",
        action="store_true",
        help="Salva resultados em CSV em results/"
    )

    # anova
    anova_parser = subparsers.add_parser(
        "anova",
        help="ANOVA de retornos diários entre criptomoedas"
    )
    anova_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="nível de significância (default=0.05)"
    )
    anova_parser.add_argument(
        "--save",
        action="store_true",
        help="salvar resultados em results/"
    )

    # anova-groups
    ag_parser = subparsers.add_parser(
        "anova-groups",
        help="ANOVA de retornos por grupos de criptomoedas"
    )
    ag_parser.add_argument(
        "--metric",
        choices=["volatility","mean_return","volume"],
        default="volatility",
        help="métrica para agrupar (default=volatility)"
    )
    ag_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="nível de significância (default=0.05)"
    )
    ag_parser.add_argument(
        "--save",
        action="store_true",
        help="salvar resultados em results/"
    )

    args = parser.parse_args()

    if args.command == "analyze":
        analyze(args)
    elif args.command == "simulate":
        simulate(args)
    elif args.command == "train":
        train(args)
    elif args.command == "compare":
        subprocess.run([
            sys.executable, "-m", "scripts.compare_models",
            "--crypto", args.crypto
        ], check=True)
    elif args.command == "hypothesis":
        cmd = [
            sys.executable, "-m", "scripts.test_hypothesis",
            "--threshold", str(args.threshold),
            "--alpha",    str(args.alpha)
        ]
        if args.save:
            cmd.append("--save")
        subprocess.run(cmd, check=True)
    elif args.command == "anova":
        cmd = [
            sys.executable, "-m", "scripts.anova",
            "--alpha", str(args.alpha)
        ] + (["--save"] if args.save else [])
        subprocess.run(cmd, check=True)
    elif args.command == "anova-groups":
        cmd = [
            sys.executable, "-m", "scripts.anova_groups",
            "--metric", str(args.metric),
            "--alpha",  str(args.alpha)
        ] + (["--save"] if args.save else [])
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

