import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas de features derivadas ao DataFrame da criptomoeda.
    """
    df = df.copy()
    df["pct_change_1d"] = df["Close"].pct_change()
    df["volume_change_1d"] = df["Volume"].pct_change()
    
    # Médias móveis
    df["ma_3"] = df["Close"].rolling(window=3).mean()
    df["ma_7"] = df["Close"].rolling(window=7).mean()
    df["ma_14"] = df["Close"].rolling(window=14).mean()

    # Desvios padrão móveis
    df["std_7"] = df["Close"].rolling(window=7).std()
    df["std_14"] = df["Close"].rolling(window=14).std()
    
    # Target: preço de fechamento do dia seguinte
    df["target"] = df["Close"].shift(-1)

    # Remove as primeiras linhas com NaN
    df = df.dropna().reset_index(drop=True)
    
    return df
