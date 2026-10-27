import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Define a raiz do projeto e ajusta o sys.path para importar a pasta src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução
from src.utils.general import get_datasets, get_protocol, get_rolling_origins

warnings.filterwarnings("ignore")

# Limite do alerta de viés: |média(previsão - real)| > fração * desvio-padrão do real.
# Um viés dessa ordem costuma indicar previsão salva na escala errada (ex.: z-score vs original).
BIAS_ALERT_THRESHOLD = 0.5

MODEL_COLUMNS = {
    "ARIMA": "ARIMA",
    "ETS": "ETS",
    "Prophet": "Prophet",
    "MLP": "MLP",
    "LSTM": "LSTM",
    "N-HiTS": "NHITS_Standalone",
    "Informer": "Informer_Standalone",
    "Híbrido (ARIMA + N-HiTS)": "Hybrid_ARIMA_NHITS",
    "Híbrido (Prophet + N-HiTS)": "Hybrid_Prophet_NHITS",
    "Ingênuo (último valor)": "Naive",
    "Ingênuo Sazonal": "SeasonalNaive",
}


def naive_forecasts(df_raw, start_ds, n_steps, season_length):
    """
    Baselines ingênuos a partir da última observação antes de `start_ds`:
    - Naive: repete o último valor observado
    - SeasonalNaive: repete o último ciclo de `season_length` passos (24 = 1 dia nos ETTh)
    Usa apenas dados anteriores à origem (sem vazamento do período avaliado).
    """
    df_sorted = df_raw.sort_values("ds").reset_index(drop=True)
    start_idx = df_sorted.index[df_sorted["ds"] == start_ds]
    if len(start_idx) == 0:
        raise ValueError(f"Origem dos baselines ingênuos não encontrada nos dados reais: {start_ds}")
    start_idx = start_idx[0]

    history = df_sorted["y_real"].iloc[:start_idx].to_numpy()
    if len(history) < season_length:
        raise ValueError(f"Histórico ({len(history)}) menor que a sazonalidade ({season_length}).")

    future_ds = df_sorted["ds"].iloc[start_idx : start_idx + n_steps].reset_index(drop=True)
    steps = np.arange(len(future_ds))
    last_cycle = history[-season_length:]

    return pd.DataFrame(
        {
            "ds": future_ds,
            "Naive": history[-1],
            "SeasonalNaive": last_cycle[steps % season_length],
        }
    )


def check_model_columns(df_eval, model_columns):
    """Avisa (sem interromper) quais colunas de modelo não estão no DataFrame de avaliação."""
    missing = [f"{name} ({col})" for name, col in model_columns.items() if col not in df_eval.columns]
    if missing:
        print(f"   ⚠️ Aviso: colunas de modelo ausentes, serão ignoradas: {', '.join(missing)}")
    return missing


def check_bias(df_eval, model_columns, reference_std, threshold=BIAS_ALERT_THRESHOLD):
    """Alerta quando o viés médio de um modelo passa de `threshold` desvios-padrão do real."""
    alerts = {}
    for name, col in model_columns.items():
        if col not in df_eval.columns:
            continue
        bias = (df_eval[col] - df_eval["y_real"]).mean()
        if abs(bias) > threshold * reference_std:
            alerts[name] = bias
            print(
                f"   🚨 Alerta de viés: {name} tem viés médio {bias:+.3f} "
                f"(> {threshold} x desvio-padrão {reference_std:.3f}). Verifique a escala da previsão."
            )
    return alerts


NORM_STATS_OPTIONS = ("global", "train")


def compute_norm_stats(df_raw, mode, train_size=None):
    """
    Média e desvio-padrão para a normalização z-score das métricas.
    - global: série inteira (inclui o período de teste)
    - train: apenas as `train_size` primeiras linhas (em ordem temporal)
    """
    if mode not in NORM_STATS_OPTIONS:
        raise ValueError(f"evaluation.norm_stats inválido: '{mode}'. Use um de {NORM_STATS_OPTIONS}.")

    if mode == "global":
        y = df_raw["y_real"]
    else:
        if not train_size:
            raise ValueError("evaluation.norm_stats='train' exige 'train_size' configurado para o dataset.")
        if train_size >= len(df_raw):
            raise ValueError(f"train_size ({train_size}) deve ser menor que o tamanho da série ({len(df_raw)}).")
        y = df_raw.sort_values("ds")["y_real"].iloc[:train_size]

    return y.mean(), y.std()


def check_no_train_overlap(df_raw, df_eval, train_size):
    """Garante que a janela avaliada começa depois do fim do bloco de treino."""
    train_end = df_raw.sort_values("ds")["ds"].iloc[train_size - 1]
    eval_start = df_eval["ds"].min()
    if eval_start <= train_end:
        raise ValueError(
            f"Janela de avaliação começa em {eval_start}, antes do fim do treino ({train_end}). "
            "A normalização 'train' exige que o teste fique depois do bloco de treino."
        )


def slice_horizon(df_eval, h):
    """Retorna os h primeiros passos e falha se não houver exatamente h linhas."""
    df_h = df_eval.head(h)
    if len(df_h) != h:
        raise ValueError(
            f"Horizonte {h}: esperadas {h} linhas após o merge, encontradas {len(df_h)}. "
            "Previsões e dados reais não estão alinhados (datas faltando ou horizonte maior que a previsão)."
        )
    return df_h


def calculate_metrics(y_true, y_pred, scaler_mean, scaler_std):
    # Aplica a Padronização Z-Score (Escala Acadêmica) antes de calcular o erro
    y_true_scaled = (y_true - scaler_mean) / scaler_std
    y_pred_scaled = (y_pred - scaler_mean) / scaler_std

    mse = np.mean((y_true_scaled - y_pred_scaled) ** 2)
    mae = np.mean(np.abs(y_true_scaled - y_pred_scaled))
    return mse, mae


# ==============================================================================
# PROTOCOLO ROLLING (várias origens no bloco de teste)
# ==============================================================================
# Pasta (chave em results_paths) -> sufixo do arquivo {dataset}_{sufixo}_rolling.csv.
# Formato longo esperado: cutoff, ds, <colunas de MODEL_COLUMNS> (unique_id e y são ignorados).
ROLLING_SOURCES = {
    "arima": "arima",
    "ets": "ets",
    "prophet": "prophet",
    "deep_learning": "deep",
    "hybrid": "hybrid",
    "hybrid_prophet": "hybrid_prophet",
}


def evaluate_rolling(df_eval, horizons, model_columns, norm_mean, norm_std):
    """
    Métricas por janela (cutoff) e horizonte, e o resumo média ± desvio entre janelas.
    Um modelo sem previsão completa numa janela fica de fora só daquela janela (ver n_janelas).
    """
    rows = []
    for cutoff, df_w in df_eval.sort_values(["cutoff", "ds"]).groupby("cutoff", sort=True):
        for h in horizons:
            df_h = slice_horizon(df_w, h)
            for model_name, col_name in model_columns.items():
                if col_name not in df_h.columns or df_h[col_name].isna().any():
                    continue
                mse, mae = calculate_metrics(df_h["y_real"], df_h[col_name], norm_mean, norm_std)
                rows.append({"cutoff": cutoff, "Horizonte": h, "Modelo": model_name, "MSE": mse, "MAE": mae})

    df_windows = pd.DataFrame(rows, columns=["cutoff", "Horizonte", "Modelo", "MSE", "MAE"])
    df_summary = (
        df_windows.groupby(["Horizonte", "Modelo"], sort=False)
        .agg(
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            n_janelas=("MSE", "size"),
        )
        .reset_index()
    )
    return df_windows, df_summary


class MetricsMultiHorizon:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos o YAML para mapear onde os resultados de cada modelo estão salvos
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.hybrid_dir = BASE_DIR / self.config["results_paths"]["hybrid"]
        self.deep_dir = BASE_DIR / self.config["results_paths"]["deep_learning"]
        self.ets_dir = BASE_DIR / self.config["results_paths"]["ets"]
        self.metrics_dir = BASE_DIR / self.config["results_paths"]["metrics"]
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        eval_config = self.config.get("evaluation", {})
        self.norm_stats = eval_config.get("norm_stats", "global")
        self.protocol = get_protocol(self.config)

    def calculate_metrics(self, y_true, y_pred, scaler_mean, scaler_std):
        return calculate_metrics(y_true, y_pred, scaler_mean, scaler_std)

    @staticmethod
    def load_actuals(info):
        """Dados reais (gabarito) em ordem temporal, com colunas ds e y_real."""
        df_raw = pd.read_csv(BASE_DIR / info["path"])
        df_raw = df_raw[[info["date_column"], info["target_column"]]].copy()
        df_raw.columns = ["ds", "y_real"]
        df_raw["ds"] = pd.to_datetime(df_raw["ds"])
        return df_raw

    def process_dataset(self, current_dataset, info):
        """Processa as métricas de um único dataset de forma isolada."""
        print(f"\n📊 Processando: {current_dataset}...")

        # Verifica se as previsões base (Híbrido) existem para este dataset
        hybrid_file = self.hybrid_dir / f"{current_dataset}_hybrid_predictions.csv"
        if not hybrid_file.exists():
            print(f"   ⏩ Pulando {current_dataset}: Arquivo base de previsões (ARIMA+NHITS) não encontrado.")
            return

        # 1. Carrega Híbrido e ARIMA
        df_preds = pd.read_csv(hybrid_file)
        df_preds["ds"] = pd.to_datetime(df_preds["ds"])

        # 2. Carrega Baselines de Deep Learning
        dl_file = self.deep_dir / f"{current_dataset}_deep_predictions.csv"
        if dl_file.exists():
            df_dl = pd.read_csv(dl_file)
            df_dl["ds"] = pd.to_datetime(df_dl["ds"])
            df_dl = df_dl.rename(columns={"NHITS": "NHITS_Standalone", "Informer": "Informer_Standalone"})
            df_preds = pd.merge(df_preds, df_dl.drop(columns=["unique_id"], errors="ignore"), on="ds", how="inner")

        # 3. Carrega Estatísticos Clássicos (ETS)
        ets_file = self.ets_dir / f"{current_dataset}_ets_predictions.csv"
        if ets_file.exists():
            df_ets = pd.read_csv(ets_file)
            df_ets["ds"] = pd.to_datetime(df_ets["ds"])
            df_preds = pd.merge(df_preds, df_ets.drop(columns=["unique_id"], errors="ignore"), on="ds", how="inner")

        # 3.5 Carrega Prophet e Híbrido Prophet
        prophet_dir = BASE_DIR / self.config["results_paths"]["prophet"]
        prophet_file = prophet_dir / f"{current_dataset}_prophet_predictions.csv"
        if prophet_file.exists():
            df_prophet = pd.read_csv(prophet_file)
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
            df_preds = pd.merge(df_preds, df_prophet.drop(columns=["unique_id"], errors="ignore"), on="ds", how="inner")

        hybrid_prophet_dir = BASE_DIR / self.config["results_paths"]["hybrid_prophet"]
        hybrid_prophet_file = hybrid_prophet_dir / f"{current_dataset}_hybrid_prophet_predictions.csv"
        if hybrid_prophet_file.exists():
            df_hp = pd.read_csv(hybrid_prophet_file)
            df_hp["ds"] = pd.to_datetime(df_hp["ds"])
            df_preds = pd.merge(df_preds, df_hp.drop(columns=["unique_id"], errors="ignore"), on="ds", how="inner")

        # 4. Carrega os Dados Reais (O Gabarito)
        df_raw = self.load_actuals(info)

        # Cruza tudo pela data exata
        df_eval = pd.merge(df_preds, df_raw, on="ds", how="inner")

        # Baselines ingênuos com a mesma origem dos demais modelos (início da janela avaliada)
        df_naive = naive_forecasts(df_raw, df_eval["ds"].min(), len(df_eval), info["seasonal_period"])
        df_eval = pd.merge(df_eval, df_naive, on="ds", how="left")

        # Estatísticas da normalização z-score (evaluation.norm_stats no main_config.yaml)
        train_size = info.get("train_size")
        norm_mean, norm_std = compute_norm_stats(df_raw, self.norm_stats, train_size)
        if self.norm_stats == "train":
            check_no_train_overlap(df_raw, df_eval, train_size)
        print(f"   > Normalização: {self.norm_stats} (média={norm_mean:.3f}, desvio={norm_std:.3f})")

        # Aqui usamos a lista de horizontes para fatiar o dataset
        horizons = info.get("forecast_horizon", [96, 192, 336, 720])
        results = []

        model_columns = MODEL_COLUMNS

        # Checagens de sanidade antes de calcular as métricas
        check_model_columns(df_eval, model_columns)
        check_bias(df_eval, model_columns, reference_std=norm_std)

        # O Loop de Fatiamento (Slice Later)
        for h in horizons:
            df_h = slice_horizon(df_eval, h)
            for model_name, col_name in model_columns.items():
                if col_name in df_h.columns:
                    mse, mae = self.calculate_metrics(df_h["y_real"], df_h[col_name], norm_mean, norm_std)
                    results.append(
                        {
                            "Horizonte": h,
                            "Modelo": model_name,
                            "MSE": round(mse, 4),
                            "MAE": round(mae, 4),
                            "Normalizacao": self.norm_stats,
                        }
                    )

        df_results = pd.DataFrame(results)
        print("-" * 60)
        print(df_results.to_string(index=False))

        out_file = self.metrics_dir / f"{current_dataset}_evaluation_complete.csv"
        df_results.to_csv(out_file, index=False)
        print(f"💾 Salvo em: {self.metrics_dir.name}/{out_file.name}")

    def process_dataset_rolling(self, current_dataset, info):
        """Protocolo rolling: avalia todas as origens do bloco de teste e resume média ± desvio."""
        print(f"\n📊 Processando (rolling): {current_dataset}...")

        df_raw = self.load_actuals(info).sort_values("ds").reset_index(drop=True)
        origins = get_rolling_origins(info, len(df_raw))
        max_h = info["max_horizon"]
        print(
            f"   > {len(origins)} janelas de {max_h} passos | primeira previsão em {df_raw['ds'].iloc[origins[0]]}, "
            f"última origem em {df_raw['ds'].iloc[origins[-1]]}"
        )

        # 1. Baselines ingênuos em cada origem (sempre disponíveis, calculados dos dados reais)
        naive_frames = []
        for start in origins:
            df_naive = naive_forecasts(df_raw, df_raw["ds"].iloc[start], max_h, info["seasonal_period"])
            df_naive.insert(0, "cutoff", df_raw["ds"].iloc[start - 1])
            naive_frames.append(df_naive)
        df_preds = pd.concat(naive_frames, ignore_index=True)

        # 2. Previsões rolling dos modelos (geradas pelos scripts de treino com protocol=rolling)
        model_cols = set(MODEL_COLUMNS.values())
        for dir_key, suffix in ROLLING_SOURCES.items():
            rolling_file = BASE_DIR / self.config["results_paths"][dir_key] / f"{current_dataset}_{suffix}_rolling.csv"
            if not rolling_file.exists():
                continue
            df_m = pd.read_csv(rolling_file, parse_dates=["cutoff", "ds"])
            if dir_key == "deep_learning":
                df_m = df_m.rename(columns={"NHITS": "NHITS_Standalone", "Informer": "Informer_Standalone"})
            # Só as colunas de modelo ainda não carregadas (ex.: o arquivo do híbrido também traz o ARIMA)
            keep = [c for c in df_m.columns if c in model_cols and c not in df_preds.columns]
            df_preds = pd.merge(df_preds, df_m[["cutoff", "ds"] + keep], on=["cutoff", "ds"], how="left")
            # Data do arquivo ajuda a notar previsões antigas (ex.: etapa que falhou nesta execução)
            modified = pd.Timestamp(rolling_file.stat().st_mtime, unit="s", tz="UTC").tz_convert(None)
            print(
                f"   > Carregado: {rolling_file.name} ({', '.join(keep) or 'nenhuma coluna nova'}) "
                f"| gerado em {modified:%Y-%m-%d %H:%M} UTC"
            )

        # 3. Gabarito
        df_eval = pd.merge(df_preds, df_raw, on="ds", how="inner")

        # 4. Normalização e checagens (mesmas do protocolo single_split)
        train_size = info.get("train_size")
        norm_mean, norm_std = compute_norm_stats(df_raw, self.norm_stats, train_size)
        if self.norm_stats == "train":
            check_no_train_overlap(df_raw, df_eval, train_size)
        print(f"   > Normalização: {self.norm_stats} (média={norm_mean:.3f}, desvio={norm_std:.3f})")

        model_columns = MODEL_COLUMNS
        check_model_columns(df_eval, model_columns)
        check_bias(df_eval, model_columns, reference_std=norm_std)

        # 5. Métricas por janela e resumo
        horizons = info.get("forecast_horizon", [96, 192, 336, 720])
        df_windows, df_summary = evaluate_rolling(df_eval, horizons, model_columns, norm_mean, norm_std)

        incomplete = df_summary.loc[df_summary["n_janelas"] < len(origins), "Modelo"].unique()
        if len(incomplete):
            print(f"   ⚠️ Aviso: modelos sem previsão em todas as {len(origins)} janelas: {', '.join(incomplete)}")

        df_windows_full = df_windows.copy()  # sem arredondamento, para os testes estatísticos
        for df in (df_windows, df_summary):
            num_cols = df.select_dtypes("number").columns.drop(["Horizonte", "n_janelas"], errors="ignore")
            df[num_cols] = df[num_cols].round(4)
            df["Normalizacao"] = self.norm_stats

        print("-" * 60)
        print(df_summary.to_string(index=False))

        out_summary = self.metrics_dir / f"{current_dataset}_evaluation_rolling.csv"
        out_windows = self.metrics_dir / f"{current_dataset}_evaluation_rolling_windows.csv"
        df_summary.to_csv(out_summary, index=False)
        df_windows.to_csv(out_windows, index=False)
        print(f"💾 Salvo em: {self.metrics_dir.name}/{out_summary.name} e {out_windows.name}")

        # 6. Testes estatísticos (Diebold-Mariano/HLN por janela e Friedman); falha aqui não invalida as métricas
        try:
            from src.metrics.statistical_tests import run_dm_for_dataset

            run_dm_for_dataset(current_dataset, info, df_windows_full, self.metrics_dir)
        except Exception as e:
            print(f"   ⚠️ Testes estatísticos não executados para {current_dataset}: {e}")

    def run(self):
        print("\n📈 [PASSO 8] Avaliação Multi-Horizonte (Escala Acadêmica Z-Score)")
        print(f"   Protocolo: {self.protocol} | Normalização: {self.norm_stats}")

        # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO
        datasets_to_run = get_datasets()

        if not datasets_to_run:
            print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando.")
            return

        # 2. ITERAÇÃO LIMPA
        for ds_name, info in datasets_to_run.items():
            try:
                if self.protocol == "rolling":
                    self.process_dataset_rolling(ds_name, info)
                else:
                    self.process_dataset(ds_name, info)
            except Exception as e:
                print(f"   ❌ Erro ao processar o dataset {ds_name}: {e}")

        if self.protocol == "rolling":
            try:
                from src.metrics.statistical_tests import run_friedman

                print("\n📐 Friedman + Iman-Davenport (todos os datasets com métricas rolling)")
                run_friedman(self.metrics_dir)
            except Exception as e:
                print(f"   ⚠️ Friedman não executado: {e}")

        print("\n✅ Avaliação concluída!\n" + "=" * 80)


if __name__ == "__main__":
    evaluator = MetricsMultiHorizon()
    evaluator.run()