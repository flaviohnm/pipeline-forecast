import pandas as pd

# ==============================================================================
# 1. CONTROLE DE EXECUÇÃO
# ==============================================================================
# Para rodar todos os datasets, use: ["ALL"]
# Para rodar datasets específicos, use: ["AirPassengers", "ETTh1", "Weather"]
EXECUTION_LIST = ["ETTh1"]


# ==============================================================================
# 2. DICIONÁRIO CENTRAL DE DATASETS
# ==============================================================================
DATASETS_INFO = {
    "AirPassengers": {
        "path": "data/AirPassengers.csv",
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
        "target_column": "Passengers",
        "date_column": "Month",
        "freq": "MS",
        "forecast_horizon": [12, 24],
        "max_horizon": 24,
        "test_size": 24,
        "val_size": 12,
        "seasonal_period": 12,
    },
    "ETTh1": {
        "path": "data/ETTh1.csv",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "h",
        "forecast_horizon": [24, 48, 168, 336, 720],
        "max_horizon": 720,
        "test_size": 2880,
        "val_size": 720,
        "seasonal_period": 24,
    },
    "ETTh2": {
        "path": "data/ETTh2.csv",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "h",
        "forecast_horizon": [24, 48, 168, 336, 720],
        "max_horizon": 720,
        "test_size": 2880,
        "val_size": 720,
        "seasonal_period": 24,
    },
    "ETTm1": {
        "path": "data/ETTm1.csv",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "t",
        "forecast_horizon": [24, 48, 96, 288, 672],
        "max_horizon": 672,
        "test_size": 2880,
        "val_size": 720,
        "seasonal_period": 96,
    },
    "ECL": {
        "path": "data/ECL.csv",
        "url": "https://huggingface.co/datasets/Time-HD-Anonymous/High_Dimensional_Time_Series/resolve/main/electricity.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "h",
        "forecast_horizon": [24, 48, 168, 336, 720, 960],
        "max_horizon": 960,
        "test_size": 2880,
        "val_size": 2160,
        "seasonal_period": 24,
    },
    "Weather": {
        "path": "data/Weather.csv",
        "url": "https://huggingface.co/datasets/Time-HD-Anonymous/High_Dimensional_Time_Series/resolve/main/weather.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "10t",
        "forecast_horizon": [24, 48, 168, 336, 720],
        "max_horizon": 720,
        "test_size": 10539,
        "val_size": 5269,
        "seasonal_period": 144,
    },
}


# ==============================================================================
# 3. FUNÇÕES UTILITÁRIAS
# ==============================================================================
def get_datasets():
    """
    Retorna os dicionários correspondentes aos datasets definidos na EXECUTION_LIST.
    """
    # Se "ALL" estiver na lista (ignorando maiúsculas/minúsculas), retorna todos
    if "ALL" in [ds.upper() for ds in EXECUTION_LIST]:
        return DATASETS_INFO

    selected_datasets = {}

    for ds in EXECUTION_LIST:
        if ds in DATASETS_INFO:
            selected_datasets[ds] = DATASETS_INFO[ds]
        else:
            print(f"⚠️ Aviso: Dataset '{ds}' não configurado em DATASETS_INFO. Ignorando...")

    return selected_datasets


def prepare_data(df, info, name):
    """
    Padroniza os dados brutos para o formato exigido pelo Nixtla (StatsForecast/NeuralForecast).
    """
    df_n = df[[info["date_column"], info["target_column"]]].copy()
    df_n.columns = ["ds", "y"]
    df_n["unique_id"] = name
    df_n["ds"] = pd.to_datetime(df_n["ds"])
    return df_n
