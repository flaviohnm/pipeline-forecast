import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

BASE_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(BASE_DIR))
from src.utils.general import prepare_data

warnings.filterwarnings("ignore")

dataset_name = os.getenv("DATASET", "ETTh1")


# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class ArimaScientificBaseline:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["arima"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.resid_dir = BASE_DIR / self.config["results_paths"]["residuals"]
        self.resid_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        dataset_env = os.getenv("DATASET", "ETTh1").strip()

        if dataset_env.upper() == "ALL":
            print("\n🚀 Modo BATCH detectado: Executando ARIMA Baseline para TODOS os datasets.")
            datasets_to_run = list(self.config["datasets"].keys())
        else:
            datasets_to_run = [dataset_env]

        print("\n🔬 [PASSO 1] Iniciando ARIMA Baseline Científico")
        print("Critério: Split no Horizonte Máximo")

        for name in datasets_to_run:
            if name not in self.config["datasets"]:
                print(f"⚠️ Dataset {name} não encontrado no main_config.yaml. Pulando...")
                continue

            info = self.config["datasets"][name]
            max_h = info.get("max_horizon", max(info["forecast_horizon"]))

            print(f"\n📊 Processando: {name} | H_max = {max_h}")

            raw_file = BASE_DIR / info["path"]
            if not raw_file.exists():
                print(f"⚠️ Arquivo bruto não encontrado para {name}: {raw_file}")
                continue

            df_raw = pd.read_csv(raw_file)
            df = prepare_data(df_raw, info, name)

            train_df = df.iloc[:-max_h]
            test_df = df.iloc[-max_h:]

            print(f"   > Treino: {len(train_df)} pontos | Teste (Blind): {len(test_df)} pontos")

            models = [AutoARIMA(season_length=info["seasonal_period"], approximation=True)]
            sf = StatsForecast(models=models, freq=info["freq"], n_jobs=-1)

            # --- MODELAGEM E PREVISÃO SIMULTÂNEA ---
            print(f"   > Ajustando ARIMA e gerando projeção recursiva para H={max_h}...")
            forecast_df = sf.forecast(df=train_df, h=max_h, fitted=True)

            # RENOMEANDO AutoARIMA para ARIMA
            forecast_df = forecast_df.rename(columns={"AutoARIMA": "ARIMA"})

            # --- SALVANDO PREVISÕES ---
            forecast_path = self.forecast_dir / f"{name}_arima_predictions.csv"
            forecast_df.to_csv(forecast_path, index=False)
            print(f"   ✅ Previsão salva em: {forecast_path.name}")

            # --- EXTRAÇÃO DE RESÍDUOS IN-SAMPLE ---
            print("   > Extraindo resíduos in-sample...")
            fitted = sf.forecast_fitted_values().reset_index()

            # Calculando o resíduo
            fitted["residual"] = fitted["y"] - fitted["AutoARIMA"]

            resid_path = self.resid_dir / f"{name}_arima_residuals.csv"
            df_res = fitted[["unique_id", "ds", "residual"]].rename(columns={"residual": "y"})
            df_res.to_csv(resid_path, index=False)
            print(f"   💾 Resíduos salvos em: {self.resid_dir.name}/{resid_path.name}")

        print("\n🏁 Passo 1 Finalizado. Resíduos prontos para a etapa Híbrida.\n")


if __name__ == "__main__":
    trainer = ArimaScientificBaseline()
    trainer.run()
