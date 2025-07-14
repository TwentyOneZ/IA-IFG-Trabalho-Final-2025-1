import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Configuração de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Caminhos dos arquivos
file_paths = {
    "BTC": "Dados/Dia/Poloniex_BTCUSDT_d.csv",
    "ETH": "Dados/Dia/Poloniex_ETHUSDT_d.csv",
    "LTC": "Dados/Dia/Poloniex_LTCUSDT_d.csv",
    "XRP": "Dados/Dia/Poloniex_XRPUSDT_d.csv",
    "BCH": "Dados/Dia/Poloniex_BCHUSDT_d.csv",
    "XMR": "Dados/Dia/Poloniex_XMRUSDT_d.csv",
    "DASH": "Dados/Dia/Poloniex_DASHUSDT_d.csv",
    "ETC": "Dados/Dia/Poloniex_ETCUSDT_d.csv",
    "BAT": "Dados/Dia/Poloniex_BATUSDT_d.csv",
    "ZRX": "Dados/Dia/Poloniex_ZRXUSDT_d.csv"
}

def load_crypto_data(filepath: str, symbol: str) -> pd.DataFrame:
    """
    Carrega um arquivo de dados de criptomoeda e prepara as colunas fundamentais.

    Parâmetros
    ----------
    filepath : str
        Caminho para o arquivo CSV de dados brutos da criptomoeda.
    symbol : str
        Código da criptomoeda (ex: 'BTC', 'ETH').

    Retorna
    -------
    pd.DataFrame
        DataFrame contendo colunas: Date, Open, High, Low, Close, Volume, Symbol.

    Lança
    -----
    FileNotFoundError
        Se o arquivo não existir no caminho fornecido.
    ValueError
        Se a coluna de volume esperada não for encontrada.
    """
    try:
        df = pd.read_csv(filepath, skiprows=1)
        df.columns = df.columns.str.strip()
        volume_col = f"Volume {symbol}"
        if volume_col not in df.columns:
            raise ValueError(f"Coluna de volume esperada '{volume_col}' não encontrada em {filepath}")
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            volume_col: 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df["Symbol"] = symbol
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
    except FileNotFoundError:
        logger.exception("Arquivo não encontrado: %s", filepath)
        raise
    except Exception:
        logger.exception("Erro ao carregar dados de %s", symbol)
        raise

# Cria subpastas para salvar gráficos
base_path = "figures"
subdirs = ["boxplot", "hist", "lineplot"]
for sub in subdirs:
    os.makedirs(os.path.join(base_path, sub), exist_ok=True)

# Carrega os dados
dataframes = {}
for symbol, path in file_paths.items():
    try:
        df = load_crypto_data(path, symbol)
        dataframes[symbol] = df
        logger.info("Dados carregados para %s", symbol)
    except Exception:
        logger.warning("Pulando %s devido a erro de carregamento", symbol)

all_data = pd.concat(dataframes.values(), ignore_index=True)

# Estatísticas resumo
stats_summary = all_data.groupby("Symbol")["Close"].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    max='max',
    skew=lambda x: stats.skew(x),
    kurt=lambda x: stats.kurtosis(x)
)
print("\nResumo estatístico por criptomoeda:\n", stats_summary)

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=all_data, x="Symbol", y="Close")
plt.title("Boxplot do Preço de Fechamento")
plt.savefig(f"{base_path}/boxplot/boxplot_close.png", dpi=150)
plt.close()

# Histogramas
for symbol, df in dataframes.items():
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Close"], bins=30, kde=True)
    plt.title(f"Histograma do Preço de Fechamento - {symbol}")
    plt.xlabel("Close")
    plt.ylabel("Frequência")
    plt.savefig(f"{base_path}/hist/hist_{symbol}.png", dpi=150)
    plt.close()

# Gráficos de linha com média, mediana e moda
for symbol, df in dataframes.items():
    df = df.copy()
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
    plt.savefig(f"{base_path}/lineplot/lineplot_{symbol}.png", dpi=150)
    plt.close()
