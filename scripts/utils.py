import logging
from typing import Dict, Union

import pandas as pd
import numpy as np
from scipy import stats

# Configuração do logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def simular_lucro_vetorizado(
    df: pd.DataFrame,
    threshold: float = 0.0,
    investimento_inicial: float = 1000.0
) -> pd.DataFrame:
    """
    Simula investimento diário com base na previsão do modelo.
    Compra (investe) somente se a previsão indicar valorização.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo colunas 'Date', 'Close' e 'target'
    threshold : float, default=0.0
        Valor mínimo para considerar que houve alta na previsão
    investimento_inicial : float, default=1000.0
        Valor inicial investido no primeiro dia

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas:
        - Data: data da operação
        - preco_hoje: preço de fechamento no dia
        - preco_amanha: preço de fechamento no dia seguinte
        - Previsao: valor previsto pelo modelo
        - sinal: 1 para compra, 0 para não operar
        - retorno_estrategia: retorno do dia na estratégia
        - retorno_hold: retorno do dia em buy & hold
        - capital_estrategia: capital acumulado pela estratégia
        - capital_hold: capital acumulado em buy & hold
    """
    try:
        df = df.copy()
        df["preco_hoje"] = df["Close"]
        df["preco_amanha"] = df["Close"].shift(-1)

        # Define o sinal: 1 se a previsão indicar alta (modelo prevê aumento), 0 caso contrário
        df["sinal"] = (df["target"] > df["preco_hoje"] + threshold).astype(int)

        # Retorno real do dia seguinte (com base no preço real, não na previsão)
        df["retorno_estrategia"] = np.where(
            df["sinal"] == 1,
            df["preco_amanha"] / df["preco_hoje"],
            1.0
        )
        df["retorno_hold"] = df["preco_amanha"] / df["preco_hoje"]

        # Cálculo do capital acumulado
        df["capital_estrategia"] = investimento_inicial * np.cumprod(df["retorno_estrategia"])
        df["capital_hold"] = investimento_inicial * np.cumprod(df["retorno_hold"])

        # Remove última linha com valores inválidos por causa do shift
        df = df.dropna(subset=["preco_amanha"]).reset_index(drop=True)

        # Organiza colunas para exportação
        result = df[[
            "Date", "preco_hoje", "preco_amanha", "target", "sinal",
            "retorno_estrategia", "retorno_hold",
            "capital_estrategia", "capital_hold"
        ]].rename(columns={
            "Date": "Data",
            "target": "Previsao"
        })
        logger.info("Simulação de lucro vetorizado concluída com sucesso")
        return result

    except Exception:
        logger.exception("Erro ao simular lucro vetorizado")
        raise


def hypothesis_test_mean_return(
    returns: np.ndarray,
    threshold: float,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool]]:
    """
    Testa H0: μ <= threshold vs H1: μ > threshold em returns diários.

    Parâmetros
    ----------
    returns : np.ndarray
        Série de retornos diários (ex: 0.01 para 1%).
    threshold : float
        Retorno médio mínimo sob H0 (em mesmas unidades de `returns`).
    alpha : float, default=0.05
        Nível de significância.

    Retorna
    -------
    dict
        't_stat'  : estatística t,
        'p_value' : p-valor (one-sided),
        'reject'  : True se rejeita H0 ao nível alpha.
    """
    try:
        returns = np.asarray(returns, dtype=float)
        returns = returns[~np.isnan(returns)]
        if len(returns) < 2:
            logger.warning("Amostra insuficiente para teste de hipótese")
            return {"t_stat": np.nan, "p_value": np.nan, "reject": False}

        # Ajusta para testar H0: média = threshold
        diffs = returns - threshold
        t_stat, p_two = stats.ttest_1samp(diffs, 0.0)
        # one-sided p-value
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        reject = p_one < alpha
        logger.info(
            "Teste de hipótese concluído: t_stat=%.4f, p_value=%.4f, reject=%s",
            t_stat, p_one, reject
        )
        return {"t_stat": t_stat, "p_value": p_one, "reject": reject}

    except Exception:
        logger.exception("Erro no teste de hipótese de retorno médio")
        return {"t_stat": np.nan, "p_value": np.nan, "reject": False}