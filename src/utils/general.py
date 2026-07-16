import pandas as pd

def prepare_data(df, info, name):
    """
    Padroniza os dados brutos para o formato exigido pelo Nixtla (StatsForecast/NeuralForecast).
    """
    df_n = df[[info["date_column"], info["target_column"]]].copy()
    df_n.columns = ["ds", "y"]
    df_n["unique_id"] = name
    df_n["ds"] = pd.to_datetime(df_n["ds"])
    return df_n