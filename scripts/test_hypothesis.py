import os
import glob
import argparse
import logging
import sys
from typing import List, Dict, Union

import pandas as pd
import numpy as np

from scripts.data_load import load_crypto_data
from scripts.features import create_features
from scripts.utils import hypothesis_test_mean_return

# configura logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    """
    Executa o teste de hipótese para retornos médios diários de várias criptomoedas.

    Avalia H0: μ <= threshold% vs H1: μ > threshold% utilizando teste t de uma amostra.
    Coleta todos os arquivos de dados em Dados/Dia, aplica o teste e exibe tabela.
    Opcionalmente salva o resultado em CSV em results/.
    """
    parser = argparse.ArgumentParser(
        description="Teste de hipótese: μ ≥ x% para retornos médios diários"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Valor x em porcentagem (ex: 1.5 para testar μ ≥ 1.5%)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nível de significância (padrão: 0.05)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Salvar resultados em CSV em results/"
    )
    args = parser.parse_args()

    threshold_decimal: float = args.threshold / 100.0
    files: List[str] = glob.glob("Dados/Dia/Poloniex_*USDT_d.csv")
    records: List[Dict[str, Union[str, float, bool]]] = []

    for path in files:
        # Extrai símbolo, ex: Poloniex_BTCUSDT_d.csv → BTC
        basename: str = os.path.basename(path)
        symbol: str = basename.split("_")[1].replace("USDT", "")

        try:
            df: pd.DataFrame = load_crypto_data(path)
            df = create_features(df)
            df.replace([pd.NA, pd.NaT], pd.NA, inplace=True)
            df.dropna(subset=["pct_change_1d"], inplace=True)

            returns: np.ndarray = df["pct_change_1d"].values
            res: Dict[str, Union[float, bool]] = hypothesis_test_mean_return(
                returns,
                threshold=threshold_decimal,
                alpha=args.alpha
            )

            records.append({
                "symbol": symbol,
                "t_stat": res["t_stat"],
                "p_value": res["p_value"],
                "reject_H0": res["reject"]
            })
            logger.info(
                "Processado %s: t_stat=%.4f, p_value=%.4f, reject_H0=%s",
                symbol, res["t_stat"], res["p_value"], res["reject"]
            )

        except Exception:
            logger.exception("Falha ao processar dados de %s", symbol)

    result_df: pd.DataFrame = pd.DataFrame(records)
    print(result_df.to_string(index=False))

    if args.save:
        try:
            os.makedirs("results", exist_ok=True)
            out_csv: str = f"results/hypothesis_{args.threshold}pct.csv"
            result_df.to_csv(out_csv, index=False)
            logger.info("Resultados salvos em %s", out_csv)
            print(f"\nResultados salvos em {out_csv}")
        except Exception:
            logger.exception("Erro ao salvar resultados em CSV")

    sys.exit(0)


if __name__ == "__main__":
    main()
