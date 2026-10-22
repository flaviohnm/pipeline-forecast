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
from src.utils.general import check_rolling_cutoffs, get_datasets, get_protocol, get_rolling_origins, prepare_data

warnings.filterwarnings("ignore")


def fit_predict_rolling(model_name, model_instance, df, info, input_size):
    """
    Protocolo rolling: treina uma única vez em treino+val (early stopping no bloco de validação)
    e prevê cada origem do bloco de teste usando só a janela de contexto anterior ao cutoff.
    Retorna o formato longo unique_id, cutoff, ds, <model_name>.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    origins = get_rolling_origins(info, len(df))
    if origins[0] < input_size:
        raise ValueError(f"Histórico antes da 1ª origem ({origins[0]}) menor que o input_size ({input_size}).")

    # Valores ausentes são aceitos só depois do último cutoff (ex.: resíduos), nunca no treino ou contexto
    df_fit = df.iloc[: origins[0]]
    if df_fit["y"].isna().any():
        raise ValueError("Valores ausentes no bloco treino+val usado para treinar o modelo.")

    nf = NeuralForecast(models=[model_instance], freq=info["freq"])
    nf.fit(df=df_fit, val_size=info["rolling"]["val_size"])

    frames = []
    for start in origins:
        context_df = df.iloc[start - input_size : start]
        if context_df["y"].isna().any():
            raise ValueError(f"Valores ausentes no contexto da janela com cutoff {context_df['ds'].iloc[-1]}.")
        preds = nf.predict(df=context_df).reset_index()
        preds["cutoff"] = context_df["ds"].iloc[-1]
        frames.append(preds[["unique_id", "cutoff", "ds", model_name]])

    del nf
    gc.collect()

    df_rolling = pd.concat(frames, ignore_index=True)
    check_rolling_cutoffs(df_rolling, df, origins, h=info["max_horizon"])
    return df_rolling


class DLBaselinesTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos o YAML para pegar caminhos de salvamento e a random_seed global
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["deep_learning"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = get_protocol(self.config)

    def build_models(self, horizon, input_size):
        """Definições dos modelos, compartilhadas pelos protocolos single_split e rolling."""
        # Removemos a trava 'limit_val_batches: 0.0' para o PyTorch poder validar corretamente
        pl_trainer_kwargs = {
            "num_sanity_val_steps": 0,
            "enable_progress_bar": True,
        }

        return {
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
                accelerator="cpu",
                early_stop_patience_steps=3,
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
                accelerator="cpu",
                early_stop_patience_steps=3,
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
                accelerator="cpu",
                early_stop_patience_steps=3,
                random_seed=self.config["random_seed"],
                num_workers_loader=0,
                **pl_trainer_kwargs,
            ),
        }

    def run_rolling(self, name, info, df, input_size):
        """Treina cada modelo uma vez e prevê todas as origens do bloco de teste."""
        origins = get_rolling_origins(info, len(df))
        print(
            f"📈 Rolling: {len(origins)} janelas de H={info['max_horizon']} | treino+val = {origins[0]} pontos "
            f"(early stopping nos últimos {info['rolling']['val_size']}) | contexto = {input_size} pontos"
        )

        df_out = None
        for model_name, model_instance in self.build_models(info["max_horizon"], input_size).items():
            print(f"\n🚀 Treinando modelo (rolling): {model_name}...")
            df_model = fit_predict_rolling(model_name, model_instance, df, info, input_size)
            df_out = df_model if df_out is None else df_out.merge(df_model, on=["unique_id", "cutoff", "ds"])
            del model_instance
            gc.collect()

        out_file = self.forecast_dir / f"{name}_deep_rolling.csv"
        df_out.to_csv(out_file, index=False)
        print(f"\n✅ Previsões rolling para {name} salvas em: {self.forecast_dir.name}/{out_file.name}")

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

            if self.protocol == "rolling":
                self.run_rolling(name, info, df, input_size)
                continue

            train_df = df.iloc[:-horizon]

            # =======================================================
            # TRAVA DE SEGURANÇA: Compatibilidade de Janela PyTorch
            # O val_size não pode ser inferior ao horizonte
            # =======================================================
            safe_val_size = info.get("val_size", horizon)
            if safe_val_size < horizon:
                safe_val_size = horizon

            # Cria um DataFrame base apenas com as datas do horizonte futuro para receber as previsões
            future_dates = df.iloc[-horizon:]["ds"].reset_index(drop=True)
            final_forecasts = pd.DataFrame({"ds": future_dates})

            print(f"📈 Horizonte: {horizon} | Tamanho de Entrada (Input Size): {input_size}")
            print(f"   > Validation Size (Protegido): {safe_val_size} pontos")

            model_definitions = self.build_models(horizon, input_size)

            # --- LOOP SEQUENCIAL COM GARBAGE COLLECTION ---
            for model_name, model_instance in model_definitions.items():
                print(f"\n🚀 Treinando modelo: {model_name}...")

                nf = NeuralForecast(models=[model_instance], freq=info["freq"])

                # Treinamento agora recebe o parâmetro val_size de forma segura
                nf.fit(df=train_df, val_size=safe_val_size)

                print(f"🔮 Prevendo os próximos {horizon} pontos com {model_name}...")

                # OTIMIZAÇÃO DE PREDICAÇÃO: Passamos apenas a janela histórica estritamente necessária
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