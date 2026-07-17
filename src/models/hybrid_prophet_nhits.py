import gc
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from prophet import Prophet

# Define a raiz do projeto e ajusta o sys.path para importar a pasta src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.utils.general import prepare_data

warnings.filterwarnings("ignore")


class HybridProphetNHITSTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["hybrid_prophet"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        # 1. Recupera o dataset ativo do ambiente
        dataset_env = os.getenv("DATASET", "ETTh1").strip()

        # 2. Configura a lista de execução (Batch ou Único)
        if dataset_env.upper() == "ALL":
            print("\n🚀 Modo BATCH detectado: Executando Híbrido (Prophet + N-HiTS) para TODOS os datasets.")
            datasets_to_run = list(self.config["datasets"].keys())
        else:
            datasets_to_run = [dataset_env]

        # 3. Itera sobre cada dataset individualmente
        for name in datasets_to_run:
            print(f"\n🧬 Iniciando Treinamento: Híbrido (Prophet + N-HiTS) - {name}")

            # Proteção contra chaves inválidas no YAML
            if name not in self.config["datasets"]:
                print(f"⚠️ Dataset {name} não encontrado no main_config.yaml. Pulando...")
                continue

            info = self.config["datasets"][name]
            raw_file = BASE_DIR / info["path"]

            # Proteção se o dataset não tiver sido baixado
            if not raw_file.exists():
                print(f"⚠️ Arquivo bruto não encontrado para {name}: {raw_file}. Pulando...")
                continue

            df_raw = pd.read_csv(raw_file)
            df = prepare_data(df_raw, info, name)

            horizon = max(info["forecast_horizon"])
            input_size = horizon * 2
            train_df = df.iloc[:-horizon]

            # ==========================================
            # 1. ETAPA ESTATÍSTICA (PROPHET)
            # ==========================================
            print("\n🚀 [Etapa 1/3] Treinando o modelo base Prophet...")
            prophet_model = Prophet()
            prophet_model.fit(train_df)

            print("🔄 Extraindo os resíduos do passado (Valor Real - Previsão Prophet)...")
            # Previsão in-sample (no próprio treino) para achar onde o Prophet errou
            in_sample_preds = prophet_model.predict(train_df)

            # Criação do DataFrame de Resíduos para a Rede Neural
            res_df = train_df.copy()
            res_df["y"] = train_df["y"].values - in_sample_preds["yhat"].values

            # ==========================================
            # 2. ETAPA DEEP LEARNING (N-HiTS)
            # ==========================================
            print("\n🧠 [Etapa 2/3] Treinando N-HiTS nos resíduos não lineares (Modo Sobrevivência CPU)...")
            pl_trainer_kwargs = {
                "num_sanity_val_steps": 0,
                "enable_progress_bar": True,
                "limit_val_batches": 0.0,  # Trava de segurança da RAM
            }

            nhits_model = NHITS(
                h=horizon,
                input_size=input_size,
                max_steps=500,
                batch_size=8,
                scaler_type="standard",
                accelerator="cpu",
                random_seed=self.config["random_seed"],
                num_workers_loader=0,
                **pl_trainer_kwargs,
            )

            nf = NeuralForecast(models=[nhits_model], freq=info["freq"])
            nf.fit(df=res_df)

            # ==========================================
            # 3. COMBINAÇÃO (PREVISÃO FUTURA)
            # ==========================================
            print("\n🔮 [Etapa 3/3] Gerando projeções futuras e combinando as matrizes...")

            # Previsão do Prophet para o futuro
            future_dates = prophet_model.make_future_dataframe(
                periods=horizon, freq=info["freq"], include_history=False
            )
            prophet_future = prophet_model.predict(future_dates)

            # Previsão do N-HiTS (usando apenas a janela de contexto para não estourar a RAM)
            context_df = res_df.tail(input_size)
            nhits_future = nf.predict(df=context_df).reset_index()

            # Soma matemática dos vetores
            final_values = prophet_future["yhat"].values + nhits_future["NHITS"].values

            # Montagem do arquivo final
            final_forecasts = pd.DataFrame(
                {"unique_id": name, "ds": prophet_future["ds"], "Hybrid_Prophet_NHITS": final_values}
            )

            out_file = self.forecast_dir / f"{name}_hybrid_prophet_predictions.csv"
            final_forecasts.to_csv(out_file, index=False)

            print("🧹 Limpando a memória...")
            del nf, nhits_model, prophet_model
            gc.collect()

            print(f"✅ Previsão Híbrida salva em: {self.forecast_dir.name}/{out_file.name}\n")


if __name__ == "__main__":
    trainer = HybridProphetNHITSTrainer()
    trainer.run()
