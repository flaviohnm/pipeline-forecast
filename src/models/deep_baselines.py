import gc
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, MLP, NHITS, Informer

# Define a raiz do projeto e o sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução e o preparador
from src.utils.general import get_datasets, prepare_data

warnings.filterwarnings("ignore")


class DLBaselinesTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos o YAML para pegar caminhos de salvamento e a random_seed global
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["deep_learning"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\n🤖 Iniciando Treinamento: Baselines de Deep Learning")

        # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO
        datasets_to_run = get_datasets()

        if not datasets_to_run:
            print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando.")
            return

        # 2. ITERAÇÃO LIMPA
        for name, info in datasets_to_run.items():
            print(f"\n=======================================")
            print(f"📊 Processando Dataset: {name}")
            print(f"=======================================")

            raw_file = BASE_DIR / info["path"]

            if not raw_file.exists():
                print(f"⚠️ Arquivo bruto não encontrado para {name}: {raw_file}. Pulando...")
                continue

            df_raw = pd.read_csv(raw_file)
            df = prepare_data(df_raw, info, name)

            # Usando o max_horizon que já vem configurado do general.py
            horizon = info["max_horizon"]
            input_size = horizon * 2  

            train_df = df.iloc[:-horizon]

            # Cria um DataFrame base apenas com as datas do horizonte futuro para receber as previsões
            future_dates = df.iloc[-horizon:]["ds"].reset_index(drop=True)
            final_forecasts = pd.DataFrame({"ds": future_dates})

            print(f"📈 Horizonte: {horizon} | Tamanho de Entrada (Input Size): {input_size}")

            pl_trainer_kwargs = {
                "num_sanity_val_steps": 0,
                "enable_progress_bar": True,
                "limit_val_batches": 0.0,
            }

            model_definitions = {
                "MLP": MLP(
                    h=horizon,
                    input_size=input_size,
                    max_steps=300,
                    batch_size=32,
                    scaler_type="standard",
                    accelerator="cpu",
                    early_stop_patience_steps=3,
                    random_seed=self.config["random_seed"],
                    num_workers_loader=0,
                    **pl_trainer_kwargs,
                ),
                "LSTM": LSTM(
                    h=horizon,
                    input_size=input_size,
                    max_steps=300,
                    batch_size=8,
                    encoder_hidden_size=64,
                    scaler_type="standard",
                    early_stop_patience_steps=3,
                    accelerator="cpu",
                    random_seed=self.config["random_seed"],
                    num_workers_loader=0,
                    **pl_trainer_kwargs,
                ),
                "NHITS": NHITS(
                    h=horizon,
                    input_size=input_size,
                    max_steps=400,
                    batch_size=32,
                    scaler_type="standard",
                    early_stop_patience_steps=3,
                    accelerator="cpu",
                    random_seed=self.config["random_seed"],
                    num_workers_loader=0,
                    **pl_trainer_kwargs,
                ),
                "Informer": Informer(
                    h=horizon,
                    input_size=horizon,
                    max_steps=300,
                    batch_size=8,
                    scaler_type="standard",
                    early_stop_patience_steps=3,
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

                # Treinamento
                nf.fit(df=train_df)

                print(f"🔮 Prevendo os próximos {horizon} pontos com {model_name}...")

                # OTIMIZAÇÃO DE PREDICAÇÃO: Passamos apenas a janela histórica
                context_df = train_df.tail(input_size)
                preds = nf.predict(df=context_df).reset_index()

                final_forecasts[model_name] = preds[model_name].values

                print(f"🧹 Limpando a memória RAM alocada pelo {model_name}...")
                del nf
                del model_instance
                del preds
                del context_df
                gc.collect()

            out_file = self.forecast_dir / f"{name}_deep_predictions.csv"

            # Adiciona a coluna unique_id no final para manter a compatibilidade
            final_forecasts.insert(0, "unique_id", name)
            final_forecasts.to_csv(out_file, index=False)

            print(f"\n✅ Todas as previsões para {name} foram salvas com segurança em: {self.forecast_dir.name}/{out_file.name}")

        print("\n🏁 Processo de Baselines de Deep Learning Finalizado!\n")


if __name__ == "__main__":
    trainer = DLBaselinesTrainer()
    trainer.run()