import gc
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from neuralforecast import NeuralForecast

# Define a raiz do projeto e o sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução e o preparador
from src.models.deep_profile import build_models, load_deep_config, model_horizons
from src.utils.compute import describe_compute
from src.utils.general import check_rolling_cutoffs, get_datasets, get_protocol, get_rolling_origins, prepare_data

warnings.filterwarnings("ignore")


def free_memory():
    """Libera RAM e, se houver GPU, a memória de vídeo entre um modelo e o próximo."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def fit_predict_rolling(model_name, model_instance, df, info, input_size, horizon=None, on_fitted=None):
    """
    Protocolo rolling: treina uma única vez em treino+val (early stopping no bloco de validação)
    e prevê cada origem do bloco de teste usando só a janela de contexto anterior ao cutoff.
    `horizon` é o comprimento H do modelo (padrão: info["max_horizon"]); as origens são sempre as mesmas,
    definidas pelo max_horizon, para que todos os modelos sejam avaliados nas mesmas janelas.
    `on_fitted(nf)` (opcional) roda logo após o treino, antes das previsões (ex.: checagens na validação).
    Retorna o formato longo unique_id, cutoff, ds, <model_name>.
    """
    horizon = horizon or info["max_horizon"]
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
    if on_fitted is not None:
        on_fitted(nf)

    frames = []
    for start in origins:
        context_df = df.iloc[start - input_size : start]
        if context_df["y"].isna().any():
            raise ValueError(f"Valores ausentes no contexto da janela com cutoff {context_df['ds'].iloc[-1]}.")
        preds = nf.predict(df=context_df).reset_index()
        preds["cutoff"] = context_df["ds"].iloc[-1]
        frames.append(preds[["unique_id", "cutoff", "ds", model_name]])

    del nf
    free_memory()

    df_rolling = pd.concat(frames, ignore_index=True)
    check_rolling_cutoffs(df_rolling, df, origins, h=horizon)
    return df_rolling


class DLBaselinesTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos o YAML para pegar caminhos de salvamento e a random_seed global
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["deep_learning"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = get_protocol(self.config)
        self.deep_cfg = load_deep_config(self.config)

    def build_models(self, horizon, input_size=None, names=None, extra_kwargs=None):
        """Modelos de um comprimento H, montados pelo perfil de hiperparâmetros (deep_learning no YAML)."""
        return build_models(horizon, self.config, input_size=input_size, names=names, extra_kwargs=extra_kwargs)

    def run_rolling(self, name, info, df):
        """
        Treina cada modelo uma vez por comprimento H (strategy: per_horizon) e prevê todas as origens do bloco de teste.
        Saída: {dataset}_deep_rolling.csv com a coluna H (comprimento do modelo) além de cutoff, ds e uma coluna por modelo.
        O arquivo é regravado ao fim de cada H, então uma falha tardia não perde o que já foi treinado.
        """
        origins = get_rolling_origins(info, len(df))
        horizons = model_horizons(info, self.deep_cfg)
        input_size = int(self.deep_cfg["input_size"])
        print(
            f"📈 Rolling: {len(origins)} janelas | estratégia: {self.deep_cfg['strategy']} | H = {horizons} | "
            f"treino+val = {origins[0]} pontos (early stopping nos últimos {info['rolling']['val_size']}) | "
            f"contexto = {input_size} pontos"
        )

        out_file = self.forecast_dir / f"{name}_deep_rolling.csv"
        per_h = []
        for h in horizons:
            df_h = None
            for model_name, model_instance in self.build_models(h, input_size).items():
                print(f"\n🚀 Treinando (rolling) H={h}: {model_name}...")
                df_model = fit_predict_rolling(model_name, model_instance, df, info, input_size, horizon=h)
                df_h = df_model if df_h is None else df_h.merge(df_model, on=["unique_id", "cutoff", "ds"])
                del model_instance
                free_memory()
            df_h.insert(1, "H", h)
            per_h.append(df_h)
            pd.concat(per_h, ignore_index=True).to_csv(out_file, index=False)
            print(f"💾 H={h} concluído; previsões parciais salvas em {out_file.name}")

        print(f"\n✅ Previsões rolling para {name} salvas em: {self.forecast_dir.name}/{out_file.name}")

    def run(self):
        print("\n🤖 Iniciando Treinamento: Baselines de Deep Learning")
        print(describe_compute(self.config))

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
            input_size = int(self.deep_cfg["input_size"])

            if self.protocol == "rolling":
                self.run_rolling(name, info, df)
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
                free_memory()

            out_file = self.forecast_dir / f"{name}_deep_predictions.csv"

            # Adiciona a coluna unique_id no final para manter a compatibilidade
            final_forecasts.insert(0, "unique_id", name)
            final_forecasts.to_csv(out_file, index=False)

            print(f"\n✅ Todas as previsões para {name} foram salvas com segurança em: {self.forecast_dir.name}/{out_file.name}")

        print("\n🏁 Processo de Baselines de Deep Learning Finalizado!\n")


if __name__ == "__main__":
    trainer = DLBaselinesTrainer()
    trainer.run()