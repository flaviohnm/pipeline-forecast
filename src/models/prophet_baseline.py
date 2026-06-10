import os
import warnings
from pathlib import Path

import pandas as pd
import yaml
from prophet import Prophet

warnings.filterwarnings("ignore")
dataset_name = os.getenv("DATASET", "ETTh1")

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class ProphetBaselineTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["prophet"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, df, info, name):
        # O Prophet exige estritamente que as colunas se chamem 'ds' e 'y'
        df_n = df[[info["date_column"], info["target_column"]]].copy()
        df_n.columns = ["ds", "y"]
        df_n["unique_id"] = name
        df_n["ds"] = pd.to_datetime(df_n["ds"])
        return df_n

    def run(self):
        print(f"\n📈 Iniciando Treinamento: Baseline (Prophet) - {dataset_name}")

        info = self.config["datasets"][dataset_name]

        raw_file = BASE_DIR / info["path"]
        df_raw = pd.read_csv(raw_file)
        df = self.prepare_data(df_raw, info, dataset_name)

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
        final_forecast = forecast[['ds', 'yhat']].rename(columns={'yhat': 'Prophet'})
        final_forecast.insert(0, "unique_id", dataset_name)

        out_file = self.forecast_dir / f"{dataset_name}_prophet_predictions.csv"
        final_forecast.to_csv(out_file, index=False)
        print(f"✅ Previsão Prophet salva em: {self.forecast_dir.name}/{out_file.name}\n")


if __name__ == "__main__":
    trainer = ProphetBaselineTrainer()
    trainer.run()