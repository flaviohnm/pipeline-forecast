import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

BASE_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(BASE_DIR))

from src.utils.general import prepare_data

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class ETSBaselineTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["ets"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        dataset_env = os.getenv("DATASET", "ETTh1").strip()

        if dataset_env.upper() == "ALL":
            print("\n🚀 Modo BATCH detectado: Executando AutoETS para TODOS os datasets.")
            datasets_to_run = list(self.config["datasets"].keys())
        else:
            datasets_to_run = [dataset_env]

        for dataset_name in datasets_to_run:
            print(f"\n📈 Iniciando Treinamento: Baseline Estatística (AutoETS) - {dataset_name}")

            # Verifica se o dataset existe no YAML para evitar KeyError
            if dataset_name not in self.config["datasets"]:
                print(f"⚠️ Dataset {dataset_name} não encontrado no main_config.yaml. Pulando...")
                continue

            info = self.config["datasets"][dataset_name]

            raw_file = BASE_DIR / info["path"]
            if not raw_file.exists():
                print(f"⚠️ Arquivo bruto não encontrado para {dataset_name}: {raw_file}")
                continue
                
            df_raw = pd.read_csv(raw_file)
            df = prepare_data(df_raw, info, dataset_name)

            horizon = max(info["forecast_horizon"])
            train_df = df.iloc[:-horizon]

            season_length = info.get("seasonal_period", 24)

            models = [AutoETS(season_length=season_length, model="ZZA")]

            sf = StatsForecast(df=train_df, models=models, freq=info["freq"], n_jobs=-1)

            print(f"🚀 Calculando AutoETS para {dataset_name} (Sazonalidade: {season_length})...")
            forecasts = sf.forecast(h=horizon)

            forecasts = forecasts.reset_index().rename(columns={"AutoETS": "ETS"})

            out_file = self.forecast_dir / f"{dataset_name}_ets_predictions.csv"
            forecasts.to_csv(out_file, index=False)
            print(f"✅ Previsão ETS salva em: {self.forecast_dir.name}/{out_file.name}")


if __name__ == "__main__":
    trainer = ETSBaselineTrainer()
    trainer.run()