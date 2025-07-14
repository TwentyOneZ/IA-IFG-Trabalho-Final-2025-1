import logging
from typing import Dict, List, Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Configuração do logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    k: int = 5
) -> Dict[str, Any]:
    """
    Treina um modelo com validação cruzada K-Fold e retorna métricas de erro.

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de features.
    y : np.ndarray
        Vetor de valores alvo.
    model : any estimator
        Instância do modelo que implementa fit() e predict().
    k : int, default=5
        Número de folds para a validação cruzada.

    Retorna
    -------
    dict
        Dicionário contendo:
        - 'model': o modelo treinado na última iteração,
        - 'rmse_mean': média dos RMSEs,
        - 'rmse_std': desvio padrão dos RMSEs,
        - 'mae_mean': média dos MAEs,
        - 'mae_std': desvio padrão dos MAEs.
    """
    try:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        rmse_list: List[float] = []
        mae_list: List[float] = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            rmse_list.append(rmse)
            mae_list.append(mae)

        metrics = {
            "model": model,
            "rmse_mean": np.mean(rmse_list),
            "rmse_std": np.std(rmse_list),
            "mae_mean": np.mean(mae_list),
            "mae_std": np.std(mae_list)
        }
        logger.info(
            "Treinamento CV concluído: RMSE médio=%.4f, MAE médio=%.4f",
            metrics["rmse_mean"], metrics["mae_mean"]
        )
        return metrics

    except Exception:
        logger.exception("Erro ao treinar modelo com validação cruzada")
        raise


def get_models(degree_list: List[int] = [2, 3]) -> Dict[str, Any]:
    """
    Retorna um dicionário com instâncias dos modelos a serem testados.

    Parâmetros
    ----------
    degree_list : list of int, default=[2,3]
        Lista de graus para gerar modelos polinomiais.

    Retorna
    -------
    dict
        Dicionário onde as chaves são nomes de modelos
        e os valores são as instâncias correspondentes ou tuplas
        ("poly", transformador_polynomial, regressão).
    """
    try:
        models: Dict[str, Any] = {
            "linear": LinearRegression(),
            "mlp": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
        }

        for d in degree_list:
            models[f"poly_deg{d}"] = (
                "poly",
                PolynomialFeatures(degree=d),
                LinearRegression()
            )

        logger.info("Modelos instanciados: %s", list(models.keys()))
        return models

    except Exception:
        logger.exception("Erro ao instanciar modelos")
        raise
