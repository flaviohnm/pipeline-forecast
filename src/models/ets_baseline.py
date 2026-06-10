import os
import warnings
from pathlib import Path

import pandas as pd
import yaml
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

warnings.filterwarnings("ignore")
dataset_name = os.getenv("DATASET", "ETTh1")

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class ETSBaselineTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["ets"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, df, info, name):
        df_n = df[[info["date_column"], info["target_column"]]].copy()
        df_n.columns = ["ds", "y"]
        df_n["unique_id"] = name
        df_n["ds"] = pd.to_datetime(df_n["ds"])
        return df_n

    def run(self):
        print(f"\n📈 Iniciando Treinamento: Baseline Estatística (AutoETS) - {dataset_name}")

        info = self.config["datasets"][dataset_name]

        raw_file = BASE_DIR / info["path"]
        df_raw = pd.read_csv(raw_file)
        df = self.prepare_data(df_raw, info, dataset_name)

        horizon = max(info["forecast_horizon"])  # 720
        train_df = df.iloc[:-horizon]

        # CORREÇÃO 2: Puxa a sazonalidade dinamicamente do YAML (Respeita 24, 96, 144...)
        season_length = info.get("seasonal_period", 24)

        # CORREÇÃO 1: Usa 'ZZA' (Erro Auto, Tendência Auto, Sazonalidade Aditiva)
        # Isso impede que a tendência exploda infinitamente no horizonte de 720 passos
        models = [AutoETS(season_length=season_length, model="ZZA")]

        sf = StatsForecast(df=train_df, models=models, freq=info["freq"], n_jobs=-1)

        print(f"\n🚀 Calculando AutoETS (Sazonalidade: {season_length})...")
        forecasts = sf.forecast(h=horizon)

        forecasts = forecasts.reset_index().rename(columns={"AutoETS": "ETS"})

        out_file = self.forecast_dir / f"{dataset_name}_ets_predictions.csv"
        forecasts.to_csv(out_file, index=False)
        print(f"✅ Previsão ETS salva em: {self.forecast_dir.name}/{out_file.name}\n")


if __name__ == "__main__":
    trainer = ETSBaselineTrainer()
    trainer.run()
