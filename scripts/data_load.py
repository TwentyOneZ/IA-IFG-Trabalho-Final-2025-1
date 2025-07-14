import pandas as pd
import logging
import os
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_crypto_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Carrega dados históricos de uma criptomoeda a partir do CSV da CryptoDataDownload.
    A coluna de volume é automaticamente identificada com base no nome do arquivo.
    
    Args:
        filepath (str): Caminho para o arquivo CSV.
    
    Returns:
        Optional[pd.DataFrame]: DataFrame com colunas padronizadas.
    """
    try:
        df = pd.read_csv(filepath, skiprows=1)
        df.columns = df.columns.str.strip()

        # Extrai a moeda base do nome do arquivo (ex: BTCUSDT -> BTC)
        filename = os.path.basename(filepath)
        symbol = filename.split("_")[1]  # BTCUSDT
        base_currency = ''.join([c for c in symbol if not c.isdigit()]).replace("USDT", "")

        volume_col = f"Volume USDT"
        if volume_col not in df.columns:
            raise ValueError(f"Coluna '{volume_col}' não encontrada em {filepath}")

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

        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logging.error(f"Erro ao carregar {filepath}: {e}")
        return None
