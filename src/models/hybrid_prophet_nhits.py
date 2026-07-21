import gc
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

# Importando o novo controlador de execução e o preparador
from src.utils.general import get_datasets, prepare_data

warnings.filterwarnings("ignore")


class HybridProphetNHITSTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos o YAML apenas para caminhos de diretórios e a random_seed global
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["hybrid_prophet"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\n🧬 Iniciando Treinamento: Híbrido (Prophet + N-HiTS)")

        # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO
        datasets_to_run = get_datasets()

        if not datasets_to_run:
            print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando.")
            return

        # 2. ITERAÇÃO LIMPA
        for name, info in datasets_to_run.items():
            print(f"\n=======================================")
            print(f"🧬 Processando Híbrido para: {name}")
            print(f"=======================================")

            raw_file = BASE_DIR / info["path"]

            # Proteção se o dataset não tiver sido baixado
            if not raw_file.exists():
                print(f"⚠️ Arquivo bruto não encontrado para {name}: {raw_file}. Pulando...")
                continue

            df_raw = pd.read_csv(raw_file)
            df = prepare_data(df_raw, info, name)

            # Usando o max_horizon que já vem configurado do general.py
            horizon = info["max_horizon"]
            input_size = horizon * 2
            train_df = df.iloc[:-horizon]

            # =======================================================
            # TRAVA DE SEGURANÇA: Compatibilidade de Janela PyTorch
            # O val_size não pode ser inferior ao horizonte
            # =======================================================
            safe_val_size = info.get("val_size", horizon)
            if safe_val_size < horizon:
                safe_val_size = horizon

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
            print("\n🧠 [Etapa 2/3] Treinando N-HiTS nos resíduos não lineares...")
            print(f"   > Validation Size (Protegido): {safe_val_size} pontos")

            pl_trainer_kwargs = {
                "num_sanity_val_steps": 0,
                "enable_progress_bar": True,
            }

            nhits_model = NHITS(
                h=horizon,
                input_size=input_size,
                max_steps=500,
                batch_size=8,
                scaler_type="standard",
                accelerator="cpu",
                early_stop_patience_steps=3,  # Requer validação
                random_seed=self.config["random_seed"],
                num_workers_loader=0,
                **pl_trainer_kwargs,
            )

            nf = NeuralForecast(models=[nhits_model], freq=info["freq"])

            # Passando o safe_val_size de forma explícita
            nf.fit(df=res_df, val_size=safe_val_size)

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
