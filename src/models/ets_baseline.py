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

        # Modificado conforme sua decisão de padronização
        self.forecast_dir = BASE_DIR / self.config["results_paths"]["ets"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, df, info, name):
        df_n = df[[info["date_column"], info["target_column"]]].copy()
        df_n.columns = ["ds", "y"]
        df_n["unique_id"] = name
        df_n["ds"] = pd.to_datetime(df_n["ds"])
        return df_n

    def run(self):
        print("\n📈 Iniciando Treinamento: Baseline Estatística (AutoETS)")

        info = self.config["datasets"][dataset_name]

        raw_file = BASE_DIR / info["path"]
        df_raw = pd.read_csv(raw_file)
        df = self.prepare_data(df_raw, info, dataset_name)

        horizon = max(info["forecast_horizon"])  # 720
        train_df = df.iloc[:-horizon]

        season_length = 24

        # Mantendo o modelo 'AAA' (Holt-Winters Aditivo) para evitar linhas retas
        models = [AutoETS(season_length=season_length, model="AAA")]

        sf = StatsForecast(df=train_df, models=models, freq=info["freq"], n_jobs=-1)

        print("\n🚀 Calculando o modelo ETS (Holt-Winters)...")
        forecasts = sf.forecast(h=horizon)

        forecasts = forecasts.reset_index().rename(columns={"AutoETS": "ETS"})

        out_file = self.forecast_dir / f"{dataset_name}_ets_predictions.csv"
        forecasts.to_csv(out_file, index=False)
        print(f"✅ Previsão ETS salva em: {self.forecast_dir.parent.name}/{self.forecast_dir.name}/{out_file.name}\n")


if __name__ == "__main__":
    trainer = ETSBaselineTrainer()
    trainer.run()
