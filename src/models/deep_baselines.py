import argparse
import gc
import sys
import traceback
import warnings
from pathlib import Path

import pandas as pd
import yaml
from neuralforecast import NeuralForecast

# Define a raiz do projeto e o sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução e o preparador
from src.models.deep_profile import MODEL_NAMES, build_models, load_deep_config, model_horizons, tuned_input_size
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
    def __init__(self, config_path="config/main_config.yaml", models=None, resume=False):
        # Mantemos o YAML para pegar caminhos de salvamento e a random_seed global
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["deep_learning"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = get_protocol(self.config)
        self.deep_cfg = load_deep_config(self.config)
        self.models = list(models or MODEL_NAMES)
        unknown = set(self.models) - set(MODEL_NAMES)
        if unknown:
            raise ValueError(f"Modelos desconhecidos: {sorted(unknown)}. Use {MODEL_NAMES}.")
        self.resume = resume
        self.failed = []

    def build_models(self, horizon, input_size=None, names=None, extra_kwargs=None, dataset=None):
        """Modelos de um comprimento H, montados pelo perfil de hiperparâmetros (deep_learning no YAML).
        Com `dataset`, usa os hiperparâmetros otimizados (hpo.py) quando existirem."""
        return build_models(horizon, self.config, input_size=input_size, names=names, extra_kwargs=extra_kwargs, dataset=dataset)

    def deep_file(self, name, model_name):
        """Arquivo de previsões de UM modelo: {dataset}_deep_{modelo}_rolling.csv (coluna H + uma coluna do modelo)."""
        return self.forecast_dir / f"{name}_deep_{model_name.lower()}_rolling.csv"

    def run_rolling_model(self, name, info, df, model_name, resume=False):
        """
        Treina UM modelo (um por comprimento H, strategy: per_horizon) e prevê todas as origens do bloco de teste.
        Saída: {dataset}_deep_{modelo}_rolling.csv com cutoff, ds, H e a coluna do modelo. O arquivo é regravado
        ao fim de cada H, então uma falha tardia não perde o que já foi treinado. Com `resume`, os H que já estão
        no arquivo são pulados (use só se a configuração não mudou desde a execução interrompida).
        """
        horizons = model_horizons(info, self.deep_cfg)
        input_size = tuned_input_size(self.config, name, model_name, int(self.deep_cfg["input_size"]))
        out_file = self.deep_file(name, model_name)

        per_h = []
        if resume and out_file.exists():
            done = pd.read_csv(out_file, parse_dates=["cutoff", "ds"])
            per_h.append(done)
            print(f"♻️ Retomando {model_name}: H já concluídos = {sorted(done['H'].unique())}")
        done_h = {int(h) for frame in per_h for h in frame["H"].unique()}

        for h in horizons:
            if h in done_h:
                continue
            print(f"\n🚀 Treinando (rolling) H={h}: {model_name}...")
            model_instance = self.build_models(h, input_size, names=[model_name], dataset=name)[model_name]
            df_model = fit_predict_rolling(model_name, model_instance, df, info, input_size, horizon=h)
            df_model.insert(1, "H", h)
            per_h.append(df_model)
            del model_instance
            free_memory()
            pd.concat(per_h, ignore_index=True).to_csv(out_file, index=False)
            print(f"💾 {model_name} H={h} concluído; previsões parciais salvas em {out_file.name}")

        print(f"\n✅ Previsões rolling de {model_name} para {name} salvas em: {self.forecast_dir.name}/{out_file.name}")

    def run_rolling(self, name, info, df, models=None, resume=False):
        """
        Roda cada modelo de forma isolada: uma falha (ex.: falta de memória) é registrada e não impede os outros.
        Retorna a lista dos modelos que falharam.
        """
        origins = get_rolling_origins(info, len(df))
        horizons = model_horizons(info, self.deep_cfg)
        input_size = int(self.deep_cfg["input_size"])
        models = list(models or MODEL_NAMES)
        print(
            f"📈 Rolling: {len(origins)} janelas | estratégia: {self.deep_cfg['strategy']} | H = {horizons} | "
            f"treino+val = {origins[0]} pontos (early stopping nos últimos {info['rolling']['val_size']}) | "
            f"contexto = {input_size} pontos | modelos = {models}"
        )

        failed = []
        for model_name in models:
            try:
                self.run_rolling_model(name, info, df, model_name, resume=resume)
            except Exception as e:  # noqa: BLE001 - isolar o modelo; o erro completo vai para o log
                traceback.print_exc()
                print(f"❌ {model_name} falhou em {name}: {type(e).__name__}: {e}")
                failed.append(model_name)
                free_memory()
        return failed

    def run(self):
        print("\n🤖 Iniciando Treinamento: Baselines de Deep Learning")
        print(describe_compute(self.config))

        # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO
        datasets_to_run = get_datasets(self.protocol)

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
                self.failed += self.run_rolling(name, info, df, models=self.models, resume=self.resume)
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
        if self.failed:
            print(f"❌ Modelos com falha: {', '.join(self.failed)}")
            return False
        return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Redes neurais isoladas (MLP, LSTM, NHITS, Informer) no protocolo rolling.")
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES),
        help="Modelos a treinar, cada um com seu próprio arquivo de previsões (padrão: todos, um após o outro).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Pula os horizontes H já presentes no arquivo do modelo (retoma uma execução interrompida).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    trainer = DLBaselinesTrainer(models=args.models, resume=args.resume)
    sys.exit(0 if trainer.run() is not False else 1)
