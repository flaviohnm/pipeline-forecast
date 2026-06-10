import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class MetricsMultiHorizon:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.hybrid_dir = BASE_DIR / self.config["results_paths"]["hybrid"]
        self.deep_dir = BASE_DIR / self.config["results_paths"]["deep_learning"]
        self.ets_dir = BASE_DIR / self.config["results_paths"]["ets"]
        self.metrics_dir = BASE_DIR / self.config["results_paths"]["metrics"]
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, y_true, y_pred, scaler_mean, scaler_std):
        # Aplica a Padronização Z-Score (Escala Acadêmica) antes de calcular o erro
        y_true_scaled = (y_true - scaler_mean) / scaler_std
        y_pred_scaled = (y_pred - scaler_mean) / scaler_std

        mse = np.mean((y_true_scaled - y_pred_scaled) ** 2)
        mae = np.mean(np.abs(y_true_scaled - y_pred_scaled))
        return mse, mae

    def process_dataset(self, current_dataset):
        """Processa as métricas de um único dataset de forma isolada."""
        print(f"\n📊 Processando: {current_dataset}...")

        # Verifica se as previsões base (Híbrido) existem para este dataset
        hybrid_file = self.hybrid_dir / f"{current_dataset}_hybrid_predictions.csv"
        if not hybrid_file.exists():
            print(f"   ⏩ Pulando {current_dataset}: Arquivo base de previsões não encontrado.")
            return

        info = self.config["datasets"][current_dataset]

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
        df_raw = pd.read_csv(BASE_DIR / info["path"])
        df_raw = df_raw[[info["date_column"], info["target_column"]]].copy()
        df_raw.columns = ["ds", "y_real"]
        df_raw["ds"] = pd.to_datetime(df_raw["ds"])

        global_mean = df_raw["y_real"].mean()
        global_std = df_raw["y_real"].std()

        # Cruza tudo pela data exata
        df_eval = pd.merge(df_preds, df_raw, on="ds", how="inner")

        horizons = info.get("forecast_horizon", [96, 192, 336, 720])
        results = []

        model_columns = {
            "ARIMA": "ARIMA",
            "ETS": "ETS",
            "Prophet": "Prophet",
            "MLP": "MLP",
            "LSTM": "LSTM",
            "N-HiTS": "NHITS_Standalone",
            "Informer": "Informer_Standalone",
            "Híbrido (ARIMA + N-HiTS)": "Hybrid_ARIMA_NHITS",
            "Híbrido (Prophet + N-HiTS)": "Hybrid_Prophet_NHITS",
        }

        for h in horizons:
            df_h = df_eval.head(h)
            for model_name, col_name in model_columns.items():
                if col_name in df_h.columns:
                    mse, mae = self.calculate_metrics(df_h["y_real"], df_h[col_name], global_mean, global_std)
                    results.append({"Horizonte": h, "Modelo": model_name, "MSE": round(mse, 4), "MAE": round(mae, 4)})

        df_results = pd.DataFrame(results)
        print("-" * 60)
        print(df_results.to_string(index=False))

        out_file = self.metrics_dir / f"{current_dataset}_evaluation_complete.csv"
        df_results.to_csv(out_file, index=False)
        print(f"💾 Salvo em: {out_file.name}")

    def run(self):
        print("\n📈 [PASSO 4] Avaliação Multi-Horizonte (Escala Acadêmica Z-Score)")

        # Lê a variável de ambiente. O padrão continua sendo ETTh1.
        target_dataset = os.getenv("METRICS", "ETTh1").strip()

        # Roteamento inteligente de execução
        if target_dataset.upper() == "ALL":
            print("🚀 Modo BATCH detectado: Executando avaliação para TODOS os datasets configurados.")
            datasets_to_run = list(self.config["datasets"].keys())
        else:
            datasets_to_run = [target_dataset]

        for ds_name in datasets_to_run:
            try:
                self.process_dataset(ds_name)
            except Exception as e:
                print(f"   ❌ Erro ao processar o dataset {ds_name}: {e}")

        print("\n✅ Avaliação concluída!\n" + "=" * 80)


if __name__ == "__main__":
    evaluator = MetricsMultiHorizon()
    evaluator.run()
