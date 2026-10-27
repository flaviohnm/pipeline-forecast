import gc
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from neuralforecast import NeuralForecast
from prophet import Prophet

# Define a raiz do projeto e ajusta o sys.path para importar a pasta src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução e o preparador
from src.models.deep_baselines import free_memory
from src.models.deep_profile import build_nhits, load_deep_config, model_horizons
from src.utils.compute import describe_compute
from src.utils.general import check_rolling_cutoffs, get_datasets, get_protocol, get_rolling_origins, prepare_data

warnings.filterwarnings("ignore")


def prophet_residuals(model, df_part):
    """Resíduos in-sample (y - yhat) do Prophet nas datas de df_part, no formato do NeuralForecast."""
    res = df_part[["unique_id", "ds"]].copy()
    res["y"] = df_part["y"].to_numpy() - model.predict(df_part[["ds"]])["yhat"].to_numpy()
    return res


def hybrid_prophet_rolling(df, info, nhits_models, input_size):
    """
    Protocolo rolling do híbrido Prophet + N-HiTS:
    1. Prophet ajustado em treino+val; cada N-HiTS é treinado UMA vez nos resíduos in-sample
       (early stopping no bloco de validação).
    2. Em cada origem, o Prophet é reajustado até o cutoff (como no baseline Prophet), os resíduos
       do contexto são recalculados com esse ajuste e Y = Prophet + N-HiTS(resíduo).
    `nhits_models`: um modelo único (comprimento = max_horizon, sem coluna H na saída) ou um dict
    {H: modelo} com um N-HiTS por comprimento de previsão (a saída ganha a coluna H). O Prophet de cada
    origem é ajustado uma vez e reaproveitado por todos os H.
    Retorna unique_id, [H,] cutoff, ds, Prophet_component, NHITS_component, Hybrid_Prophet_NHITS.
    """
    single = not isinstance(nhits_models, dict)
    models = {info["max_horizon"]: nhits_models} if single else dict(nhits_models)

    df = df.sort_values("ds").reset_index(drop=True)
    origins = get_rolling_origins(info, len(df))
    max_h = info["max_horizon"]
    if origins[0] < input_size:
        raise ValueError(f"Histórico antes da 1ª origem ({origins[0]}) menor que o input_size ({input_size}).")

    fit_df = df.iloc[: origins[0]]
    print(f"   > Ajustando Prophet em treino+val ({len(fit_df)} pontos) para extrair os resíduos de treino...")
    prophet_base = Prophet().fit(fit_df[["ds", "y"]])
    res_df = prophet_residuals(prophet_base, fit_df)

    fitted = {}
    for h, model in models.items():
        print(f"   > Treinando N-HiTS H={h} nos resíduos (validação = últimos {info['rolling']['val_size']} pontos)...")
        nf = NeuralForecast(models=[model], freq=info["freq"])
        nf.fit(df=res_df, val_size=info["rolling"]["val_size"])
        fitted[h] = nf
        free_memory()

    frames = []
    for i, start in enumerate(origins, start=1):
        train_df = df.iloc[:start]
        print(f"   > [{i}/{len(origins)}] Origem após {train_df['ds'].iloc[-1]}")
        prophet_model = prophet_base if start == origins[0] else Prophet().fit(train_df[["ds", "y"]])

        context_res = prophet_residuals(prophet_model, train_df.tail(input_size))
        future = prophet_model.make_future_dataframe(periods=max_h, freq=info["freq"], include_history=False)
        prophet_future = prophet_model.predict(future).sort_values("ds")

        for h, nf in fitted.items():
            nhits_future = nf.predict(df=context_res).reset_index().sort_values("ds")
            prophet_h = prophet_future.head(h)
            if len(nhits_future) != h or not (prophet_h["ds"].to_numpy() == nhits_future["ds"].to_numpy()).all():
                raise ValueError(f"Datas do Prophet e do N-HiTS (H={h}) divergem na origem após {train_df['ds'].iloc[-1]}.")

            frame = pd.DataFrame(
                {
                    "unique_id": df["unique_id"].iloc[0],
                    "H": h,
                    "cutoff": train_df["ds"].iloc[-1],
                    "ds": prophet_h["ds"].to_numpy(),
                    "Prophet_component": prophet_h["yhat"].to_numpy(),
                    "NHITS_component": nhits_future["NHITS"].to_numpy(),
                }
            )
            frames.append(frame)

    del fitted
    free_memory()

    df_rolling = pd.concat(frames, ignore_index=True)
    df_rolling["Hybrid_Prophet_NHITS"] = df_rolling["Prophet_component"] + df_rolling["NHITS_component"]
    for h, df_h in df_rolling.groupby("H"):
        check_rolling_cutoffs(df_h, df, origins, h=h)
    if single:
        df_rolling = df_rolling.drop(columns="H")
    return df_rolling


class HybridProphetNHITSTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos o YAML apenas para caminhos de diretórios e a random_seed global
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["hybrid_prophet"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = get_protocol(self.config)
        self.deep_cfg = load_deep_config(self.config)

    def run(self):
        print("\n🧬 Iniciando Treinamento: Híbrido (Prophet + N-HiTS)")
        print(describe_compute(self.config))

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
            input_size = int(self.deep_cfg["input_size"])

            if self.protocol == "rolling":
                horizons = model_horizons(info, self.deep_cfg)
                print(f"   > Estratégia {self.deep_cfg['strategy']}: um N-HiTS por H = {horizons}")
                nhits_models = {h: build_nhits(h, self.config, input_size) for h in horizons}
                df_rolling = hybrid_prophet_rolling(df, info, nhits_models, input_size)
                out_file = self.forecast_dir / f"{name}_hybrid_prophet_rolling.csv"
                df_rolling.to_csv(out_file, index=False)
                print(f"✅ Previsão rolling do híbrido salva em: {self.forecast_dir.name}/{out_file.name}\n")
                continue

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

            nhits_model = build_nhits(horizon, self.config, input_size)

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
