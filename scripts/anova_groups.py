import os
import glob
import argparse
import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scripts.data_load import load_crypto_data
from scripts.features import create_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ANOVA de retornos diários entre grupos de criptomoedas"
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="nível de significância")
    parser.add_argument("--metric", choices=["volatility", "mean_return", "volume"],
                        default="volatility",
                        help="como agrupar: volatilidade, retorno médio ou volume")
    parser.add_argument("--save", action="store_true", help="salvar resultados e figuras")
    args = parser.parse_args()

    files = glob.glob("Dados/Dia/Poloniex_*USDT_d.csv")
    frames = []

    logging.info(f"Processando {len(files)} criptomoedas...")

    for path in files:
        sym = os.path.basename(path).split("_")[1].replace("USDT", "")
        df = load_crypto_data(path)
        if df is None:
            continue
        df = create_features(df).dropna(subset=["pct_change_1d"])
        df["coin"] = sym
        df["retorno_pct"] = df["pct_change_1d"] * 100

        if args.metric == "volatility":
            df["metric"] = df["pct_change_1d"].rolling(window=30).std()
        elif args.metric == "mean_return":
            df["metric"] = df["pct_change_1d"].rolling(window=30).mean()
        else:  # volume
            df["metric"] = df["Volume"].rolling(window=30).mean()

        df.dropna(subset=["metric"], inplace=True)
        frames.append(df[["coin", "retorno_pct", "metric"]])

    if not frames:
        logging.error("Nenhum dado válido encontrado.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    # Agrupa por tercis da métrica
    df_all["grupo"] = pd.qcut(df_all["metric"], q=3, labels=["Low", "Mid", "High"])

    logging.info("Realizando ANOVA entre grupos baseados em '%s'...", args.metric)

    model = ols('retorno_pct ~ C(grupo)', data=df_all).fit()
    anova_result = anova_lm(model)

    f_stat = anova_result.loc["C(grupo)", "F"]
    p_val = anova_result.loc["C(grupo)", "PR(>F)"]
    logging.info(f"ANOVA — F = {f_stat:.4f}, p = {p_val:.4e}")

    if p_val < args.alpha:
        logging.info("→ p < alpha: diferenças significativas entre grupos detectadas.")

        tukey = pairwise_tukeyhsd(df_all["retorno_pct"], df_all["grupo"])
        logging.info("Teste post hoc Tukey HSD:\n%s", tukey.summary())

        if args.save:
            os.makedirs("results", exist_ok=True)
            df_all[["coin", "retorno_pct", "grupo"]].to_csv("results/dados_anova.csv", index=False)
            anova_result.to_csv("results/anova_tabela.csv")
            pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0]) \
                .to_csv("results/tukey_hsd.csv", index=False)

            # Gráfico boxplot
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df_all, x="grupo", y="retorno_pct", order=["Low", "Mid", "High"])
            plt.title(f"Retorno percentual diário por grupo ({args.metric})")
            plt.xlabel("Grupo")
            plt.ylabel("Retorno (%)")
            plt.grid(True)

            # Cálculo dos limites ignorando outliers
            q1 = df_all["retorno_pct"].quantile(0.25)
            q3 = df_all["retorno_pct"].quantile(0.75)
            iqr = q3 - q1
            lower_whisker = q1 - 1.5 * iqr
            upper_whisker = q3 + 1.5 * iqr

            # Aplicar margens de 10%
            ylim_min = lower_whisker * 0.9 if lower_whisker < 0 else 0
            ylim_max = upper_whisker * 1.1

            plt.ylim(ylim_min, ylim_max)

            plt.tight_layout()
            plt.savefig("results/anova_boxplot.png", dpi=150)
            plt.close()


            logging.info("Resultados e gráfico salvos na pasta 'results/'")
    else:
        logging.info("→ p ≥ alpha: sem evidência significativa de diferença entre grupos.")

if __name__ == "__main__":
    main()
