import gc
import os
import warnings
from pathlib import Path

import pandas as pd
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, MLP, NHITS, Informer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class DLBaselinesTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["deep_learning"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, df, info, name):
        df_n = df[[info["date_column"], info["target_column"]]].copy()
        df_n.columns = ["ds", "y"]
        df_n["unique_id"] = name
        df_n["ds"] = pd.to_datetime(df_n["ds"])
        return df_n

    def run(self):
        print("\n🤖 Iniciando Treinamento: Baselines de Deep Learning")

        dataset_name = os.getenv("DATASET", "ETTh1")
        info = self.config["datasets"][dataset_name]

        raw_file = BASE_DIR / info["path"]
        df_raw = pd.read_csv(raw_file)
        df = self.prepare_data(df_raw, info, dataset_name)

        horizon = max(info["forecast_horizon"])  # 720
        input_size = horizon * 2  # 1440

        train_df = df.iloc[:-horizon]

        # Cria um DataFrame base apenas com as datas do horizonte futuro para receber as previsões
        future_dates = df.iloc[-horizon:]["ds"].reset_index(drop=True)
        final_forecasts = pd.DataFrame({"ds": future_dates})

        print(f"\n📊 Dataset: {dataset_name} | Horizonte: {horizon}")

        pl_trainer_kwargs = {
            "num_sanity_val_steps": 0,
            "enable_progress_bar": True,
            "limit_val_batches": 0.0,
        }

        model_definitions = {
            "MLP": MLP(
                h=horizon,
                input_size=input_size,
                max_steps=500,
                batch_size=8,
                scaler_type="standard",
                accelerator="cpu",
                random_seed=self.config["random_seed"],
                num_workers_loader=0,
                **pl_trainer_kwargs,
            ),
            "LSTM": LSTM(
                h=horizon,
                input_size=input_size,
                max_steps=500,
                batch_size=1,
                encoder_hidden_size=64,
                scaler_type="standard",
                accelerator="cpu",
                random_seed=self.config["random_seed"],
                num_workers_loader=0,
                **pl_trainer_kwargs,
            ),
            "NHITS": NHITS(
                h=horizon,
                input_size=input_size,
                max_steps=500,
                batch_size=8,
                scaler_type="standard",
                accelerator="cpu",
                random_seed=self.config["random_seed"],
                num_workers_loader=0,
                **pl_trainer_kwargs,
            ),
            "Informer": Informer(
                h=horizon,
                input_size=horizon,
                max_steps=300,
                batch_size=1,
                scaler_type="standard",
                accelerator="cpu",
                random_seed=self.config["random_seed"],
                num_workers_loader=0,
                **pl_trainer_kwargs,
            ),
        }

        # --- LOOP SEQUENCIAL COM GARBAGE COLLECTION ---
        for model_name, model_instance in model_definitions.items():
            print(f"\n🚀 Treinando modelo: {model_name}...")

            nf = NeuralForecast(models=[model_instance], freq=info["freq"])

            # Treinamento: Removemos o val_size para o PyTorch não tentar validar e explodir a RAM
            nf.fit(df=train_df)

            print(f"🔮 Prevendo os próximos {horizon} pontos com {model_name}...")

            # OTIMIZAÇÃO DE PREDICAO: Passamos apenas a janela histórica estritamente necessária (input_size)
            # Isso impede que o PyTorch tente carregar milhares de linhas na memória RAM durante a inferência.
            # O input_size é 2x o horizonte. Se h=720, passamos os últimos 1440 pontos.
            context_df = train_df.tail(input_size)

            # Realizamos a previsão apenas com o contexto enxuto
            preds = nf.predict(df=context_df).reset_index()

            final_forecasts[model_name] = preds[model_name].values

            print(f"🧹 Limpando a memória RAM alocada pelo {model_name}...")
            del nf
            del model_instance
            del preds
            del context_df
            gc.collect()

        out_file = self.forecast_dir / f"{dataset_name}_deep_predictions.csv"

        # Adiciona a coluna unique_id no final para manter a compatibilidade
        final_forecasts.insert(0, "unique_id", dataset_name)
        final_forecasts.to_csv(out_file, index=False)

        print(f"\n✅ Todas as previsões salvas com segurança em: {self.forecast_dir.name}/{out_file.name}\n")


if __name__ == "__main__":
    trainer = DLBaselinesTrainer()
    trainer.run()
