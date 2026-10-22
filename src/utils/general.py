from pathlib import Path

import pandas as pd
import yaml

# Define a raiz do projeto para localizar o main_config.yaml
BASE_DIR = Path(__file__).resolve().parent.parent.parent

with open(BASE_DIR / "config" / "main_config.yaml", "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

# Bases das camadas de dados, centralizadas no main_config.yaml
RAW_DIR = _config["data_paths"]["raw"]
BRONZE_DIR = _config["data_paths"]["bronze"]

# ==============================================================================
# 1. CONTROLE DE EXECUÇÃO
# ==============================================================================
# Para rodar todos os datasets, use: ["ALL"]
# Para rodar datasets específicos, use: ["AirPassengers", "ETTh1", "Weather"]
EXECUTION_LIST = ["ETTh1"]


# ==============================================================================
# 2. DICIONÁRIO CENTRAL DE DATASETS
# ==============================================================================
# "filename" é combinado com RAW_DIR/BRONZE_DIR (vindos do main_config.yaml)
# em get_datasets() para montar "raw_path" e "path" dinamicamente.
DATASETS_INFO = {
    "AirPassengers": {
        "filename": "AirPassengers.csv",
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
        "filename": "ETTh1.csv",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "h",
        "forecast_horizon": [24, 48, 168, 336, 720],
        "max_horizon": 720,
        "test_size": 2880,
        # 12 meses de treino (protocolo padrão ETT: 12/4/4 meses)
        "train_size": 8640,
        "val_size": 720,
        "seasonal_period": 24,
    },
    "ETTh2": {
        "filename": "ETTh2.csv",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "h",
        "forecast_horizon": [24, 48, 168, 336, 720],
        "max_horizon": 720,
        "test_size": 2880,
        # 12 meses de treino (protocolo padrão ETT: 12/4/4 meses)
        "train_size": 8640,
        "val_size": 720,
        "seasonal_period": 24,
    },
    "ETTm1": {
        "filename": "ETTm1.csv",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "t",
        "forecast_horizon": [24, 48, 96, 288, 672],
        "max_horizon": 672,
        "test_size": 2880,
        # 12 meses de treino (protocolo padrão ETT: 12/4/4 meses)
        "train_size": 34560,
        "val_size": 720,
        "seasonal_period": 96,
    },
    "ECL": {
        "filename": "ECL.csv",
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
        "filename": "Weather.csv",
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
def _with_computed_paths(info):
    """Combina RAW_DIR/BRONZE_DIR (main_config.yaml) com o filename do dataset."""
    enriched = dict(info)
    enriched["raw_path"] = f"{RAW_DIR}{info['filename']}"
    enriched["path"] = f"{BRONZE_DIR}{info['filename']}"
    return enriched


def get_datasets():
    """
    Retorna os dicionários correspondentes aos datasets definidos na EXECUTION_LIST,
    já com "raw_path" e "path" calculados a partir do main_config.yaml.
    """
    # Se "ALL" estiver na lista (ignorando maiúsculas/minúsculas), retorna todos
    if "ALL" in [ds.upper() for ds in EXECUTION_LIST]:
        return {name: _with_computed_paths(info) for name, info in DATASETS_INFO.items()}

    selected_datasets = {}

    for ds in EXECUTION_LIST:
        if ds in DATASETS_INFO:
            selected_datasets[ds] = _with_computed_paths(DATASETS_INFO[ds])
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