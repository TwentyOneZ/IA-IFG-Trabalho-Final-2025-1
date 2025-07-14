import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scripts.data_load import load_crypto_data
from scripts.features import create_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ANOVA de retornos diários entre criptomoedas + Tukey HSD"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nível de significância para ANOVA (default=0.05)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Salvar resultados em CSV e imagem na pasta results/"
    )
    args = parser.parse_args()

    logging.info("Carregando dados das criptomoedas...")

    files = glob.glob("Dados/Dia/Poloniex_*USDT_d.csv")
    frames = []

    for path in files:
        simbolo = os.path.basename(path).split("_")[1].replace("USDT", "")
        df = load_crypto_data(path)
        if df is None:
            continue

        df = create_features(df).dropna(subset=["pct_change_1d"])
        df["retorno_pct"] = df["pct_change_1d"] * 100
        df["coin"] = simbolo
        frames.append(df[["coin", "retorno_pct"]])

    if not frames:
        logging.error("Nenhum dado válido encontrado.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    # ANOVA com ols
    logging.info("Executando ANOVA entre criptomoedas...")
    model = ols('retorno_pct ~ C(coin)', data=df_all).fit()
    anova_result = anova_lm(model)

    f_stat = anova_result.loc['C(coin)', 'F']
    p_val = anova_result.loc['C(coin)', 'PR(>F)']

    logging.info(f"ANOVA: F = {f_stat:.4f}, p = {p_val:.4e}")
    significativo = p_val < args.alpha

    if significativo:
        logging.info("→ p < alpha: diferenças significativas entre criptomoedas detectadas.")
        tukey = pairwise_tukeyhsd(df_all["retorno_pct"], df_all["coin"])
        logging.info("Post-hoc Tukey HSD:\n%s", tukey.summary())
    else:
        logging.info("→ p ≥ alpha: não há evidência de diferença significativa.")

    if args.save:
        os.makedirs("results", exist_ok=True)

        # Exporta dados
        df_all.to_csv("results/dados_anova_criptos.csv", index=False)
        anova_result.to_csv("results/anova_criptos.csv")
        if significativo:
            pd.DataFrame(data=tukey._results_table.data[1:], 
                         columns=tukey._results_table.data[0]) \
              .to_csv("results/tukey_criptos.csv", index=False)

            # Gera boxplot
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df_all, x="coin", y="retorno_pct")
            plt.title("Retorno percentual diário por criptomoeda")
            plt.xlabel("Criptomoeda")
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
            plt.savefig("results/boxplot_criptos.png", dpi=150)
            plt.close()


        logging.info("Resultados salvos em: results/")

if __name__ == "__main__":
    main()