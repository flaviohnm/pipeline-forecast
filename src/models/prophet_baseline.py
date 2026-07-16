import os
import warnings
from pathlib import Path
import sys

import pandas as pd
import yaml
from prophet import Prophet


BASE_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(BASE_DIR))

from src.utils.general import prepare_data

warnings.filterwarnings("ignore")
dataset_name = os.getenv("DATASET", "ETTh1")

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class ProphetBaselineTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["prophet"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        # 1. Pega o valor do .env
        dataset_env = os.getenv("DATASET", "ETTh1").strip()

        # 2. Se for ALL, monta a lista com as chaves reais do YAML. Caso contrário, usa apenas o selecionado.
        if dataset_env.upper() == "ALL":
            print("\n🚀 Modo BATCH detectado: Executando Prophet para TODOS os datasets.")
            datasets_to_run = list(self.config["datasets"].keys())
        else:
            datasets_to_run = [dataset_env]

        # 3. Itera sobre a lista de datasets válidos
        for dataset_name in datasets_to_run:
            print(f"\n🔮 Iniciando Treinamento: Baseline (Prophet) - {dataset_name}")

            # 4. Proteção contra chaves inválidas (caso 'ALL' ou outra string entre na lista por engano)
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

        print("\n🚀 Treinando o modelo Prophet...")
        # O Prophet descobre a sazonalidade automaticamente por padrão
        model = Prophet()
        model.fit(train_df)

        print(f"🔮 Prevendo os próximos {horizon} pontos com Prophet...")
        # Cria um dataframe apenas com as datas futuras
        future = model.make_future_dataframe(periods=horizon, freq=info["freq"], include_history=False)
        forecast = model.predict(future)

        # Formata a saída para o padrão do nosso avaliador
        final_forecast = forecast[["ds", "yhat"]].rename(columns={"yhat": "Prophet"})
        final_forecast.insert(0, "unique_id", dataset_name)

        out_file = self.forecast_dir / f"{dataset_name}_prophet_predictions.csv"
        final_forecast.to_csv(out_file, index=False)
        print(f"✅ Previsão Prophet salva em: {self.forecast_dir.name}/{out_file.name}\n")


if __name__ == "__main__":
    trainer = ProphetBaselineTrainer()
    trainer.run()
