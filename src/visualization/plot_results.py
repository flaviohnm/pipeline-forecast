import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as plt_sns
import yaml

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.utils.general import get_protocol


class ResultsPlotter:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = BASE_DIR / self.config["data_paths"]["raw"]
        self.metrics_dir = BASE_DIR / self.config["results_paths"]["metrics"]
        self.plots_dir = BASE_DIR / self.config["results_paths"]["plots"]
        self.forecast_hybrid = BASE_DIR / self.config["results_paths"]["forecasts"] / "hybrid"
        self.protocol = get_protocol(self.config)

        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Configuração visual padrão (Estilo Artigo Científico)
        plt.style.use("seaborn-v0_8-whitegrid")
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    def plot_mse_benchmark(self, dataset_name):
        """Gera um gráfico de barras comparando o MSE por horizonte"""
        print(f"   📊 Gerando gráfico de barras de Benchmarking para {dataset_name}...")

        metrics_file = self.metrics_dir / f"{dataset_name}_evaluation_complete.csv"
        if not metrics_file.exists():
            print(f"   ⚠️ Arquivo de métricas não encontrado: {metrics_file}")
            return

        df = pd.read_csv(metrics_file)

        plt.figure(figsize=(12, 6))
        ax = plt_sns.barplot(data=df, x="Horizonte", y="MSE", hue="Modelo", palette="viridis")

        plt.title(f"Comparação de MSE por Horizonte de Previsão ({dataset_name})", fontsize=14, fontweight="bold")
        plt.xlabel("Horizonte de Previsão (Horas)", fontsize=12)
        plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
        plt.legend(title="Modelos", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        out_path = self.plots_dir / f"{dataset_name}_mse_benchmark.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Gráfico de barras salvo em: {out_path.name}")

    def plot_time_series_forecast(self, dataset_name):
        """Gera um gráfico de linhas mostrando a previsão final vs dados reais"""
        print(f"   📈 Gerando gráfico de linhas da previsão para {dataset_name}...")

        preds_file = self.forecast_hybrid / f"{dataset_name}_hybrid_predictions.csv"
        # Convenção dos datasets ETT (ETTh1, ETTh2, ETTm1, ETTm2): coluna de data
        # sempre "date" e alvo sempre "OT". Caminho montado a partir de data_paths.raw.
        raw_file = self.raw_dir / f"{dataset_name}.csv"
        date_column = "date"
        target_column = "OT"

        if not preds_file.exists() or not raw_file.exists():
            print("   ⚠️ Arquivos de previsão ou dados reais não encontrados.")
            return

        # Carrega dados reais e previsões do Híbrido e ARIMA
        df_preds = pd.read_csv(preds_file)
        df_preds["ds"] = pd.to_datetime(df_preds["ds"])

        df_raw = pd.read_csv(raw_file)
        df_raw = df_raw[[date_column, target_column]].copy()
        df_raw.columns = ["ds", "Real"]
        df_raw["ds"] = pd.to_datetime(df_raw["ds"])

        # Junta os dados
        df_plot = pd.merge(df_preds, df_raw, on="ds", how="inner")

        # Vamos focar apenas nos primeiros 192 pontos do período de teste para o gráfico não ficar confuso
        zoom_horizon = 192
        df_plot = df_plot.head(zoom_horizon)

        plt.figure(figsize=(14, 6))

        # Plota os dados reais
        plt.plot(df_plot["ds"], df_plot["Real"], label="Real (Gabarito)", color="black", linewidth=2)

        # Plota o ARIMA (Baseline)
        plt.plot(df_plot["ds"], df_plot["ARIMA"], label="ARIMA (Linear)", color="orange", linestyle="--", alpha=0.8)

        # Plota o Híbrido (Proposto)
        plt.plot(
            df_plot["ds"],
            df_plot["Hybrid_ARIMA_NHITS"],
            label="Híbrido (Proposto)",
            color="blue",
            linewidth=2,
            alpha=0.9,
        )

        plt.title(
            f"Previsão de Curto/Médio Prazo: Real vs Modelos ({dataset_name} - Primeiras {zoom_horizon}h do teste)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Data/Hora", fontsize=12)
        plt.ylabel("Valor da Série", fontsize=12)
        plt.legend(loc="best", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = self.plots_dir / f"{dataset_name}_forecast_lines.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"   ✅ Gráfico de linhas salvo em: {out_path.name}")

    # ==========================================================================
    # PROTOCOLO ROLLING
    # ==========================================================================
    def plot_mse_benchmark_rolling(self, dataset_name):
        """Barras com o MSE médio entre janelas por horizonte e barras de erro de ±1 desvio-padrão."""
        print(f"   📊 Gerando gráfico de barras (rolling) para {dataset_name}...")

        metrics_file = self.metrics_dir / f"{dataset_name}_evaluation_rolling.csv"
        if not metrics_file.exists():
            print(f"   ⚠️ Arquivo de métricas rolling não encontrado: {metrics_file}")
            return

        df = pd.read_csv(metrics_file)
        horizons = list(dict.fromkeys(df["Horizonte"]))
        models = list(dict.fromkeys(df["Modelo"]))
        n_windows = int(df["n_janelas"].max())

        x = np.arange(len(horizons))
        width = 0.8 / len(models)
        palette = plt_sns.color_palette("viridis", len(models))

        plt.figure(figsize=(14, 6))
        for i, model in enumerate(models):
            sub = df[df["Modelo"] == model].set_index("Horizonte").reindex(horizons)
            plt.bar(
                x + (i - (len(models) - 1) / 2) * width,
                sub["MSE_mean"],
                width,
                yerr=sub["MSE_std"],
                capsize=2,
                label=model,
                color=palette[i],
                error_kw={"elinewidth": 0.8},
            )

        norm = df["Normalizacao"].iloc[0] if "Normalizacao" in df.columns else "?"
        plt.title(
            f"MSE médio ± desvio-padrão entre {n_windows} janelas ({dataset_name}, normalização: {norm})",
            fontsize=14,
            fontweight="bold",
        )
        plt.xticks(x, horizons)
        plt.xlabel("Horizonte de Previsão (passos)", fontsize=12)
        plt.ylabel("MSE (z-score)", fontsize=12)
        plt.legend(title="Modelos", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        out_path = self.plots_dir / f"{dataset_name}_mse_benchmark_rolling.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Gráfico de barras salvo em: {out_path.name}")

    def plot_time_series_forecast_rolling(self, dataset_name, window=0, zoom_horizon=192):
        """Real vs ARIMA vs Híbrido nos primeiros `zoom_horizon` passos de uma janela do protocolo rolling."""
        print(f"   📈 Gerando gráfico de linhas (rolling) para {dataset_name}...")

        preds_file = self.forecast_hybrid / f"{dataset_name}_hybrid_rolling.csv"
        raw_file = self.raw_dir / f"{dataset_name}.csv"
        if not preds_file.exists() or not raw_file.exists():
            print("   ⚠️ Arquivos de previsão rolling ou dados reais não encontrados.")
            return

        df_preds = pd.read_csv(preds_file, parse_dates=["cutoff", "ds"])
        cutoffs = sorted(df_preds["cutoff"].unique())
        cutoff = cutoffs[window]
        df_win = df_preds[df_preds["cutoff"] == cutoff].sort_values("ds")

        # Mesma convenção ETT do gráfico single_split: coluna "date" e alvo "OT"
        df_raw = pd.read_csv(raw_file)[["date", "OT"]]
        df_raw.columns = ["ds", "Real"]
        df_raw["ds"] = pd.to_datetime(df_raw["ds"])
        df_plot = df_win.merge(df_raw, on="ds", how="inner").head(zoom_horizon)

        plt.figure(figsize=(14, 6))
        plt.plot(df_plot["ds"], df_plot["Real"], label="Real (Gabarito)", color="black", linewidth=2)
        plt.plot(df_plot["ds"], df_plot["ARIMA"], label="ARIMA (Linear)", color="orange", linestyle="--", alpha=0.8)
        plt.plot(
            df_plot["ds"], df_plot["Hybrid_ARIMA_NHITS"], label="Híbrido (Proposto)", color="blue", linewidth=2, alpha=0.9
        )
        plt.title(
            f"Real vs Modelos ({dataset_name} - janela {window + 1} de {len(cutoffs)}, "
            f"origem após {pd.Timestamp(cutoff):%Y-%m-%d %H:%M}, primeiras {zoom_horizon}h)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Data/Hora", fontsize=12)
        plt.ylabel("Valor da Série", fontsize=12)
        plt.legend(loc="best", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = self.plots_dir / f"{dataset_name}_forecast_lines_rolling.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"   ✅ Gráfico de linhas salvo em: {out_path.name}")

    def run(self):
        print("\n🎨 [PASSO 5] Gerando Visualizações para o Artigo Científico")
        print(f"   Protocolo: {self.protocol}")
        rolling = self.protocol == "rolling"
        metrics_name = "evaluation_rolling" if rolling else "evaluation_complete"
        hybrid_name = "hybrid_rolling" if rolling else "hybrid_predictions"

        # Puxa a variável de ambiente (ou ETTh1 por padrão se estiver vazio)
        target_dataset = os.getenv("DATASET", "ETTh1").strip()

        if target_dataset.upper() == "ALL":
            print("🚀 Modo BATCH detectado: Filtrando datasets com dados PRONTOS para plotagem...")
            datasets_to_run = []

            # Só adiciona na fila se os arquivos já existirem nas pastas
            for raw_csv in self.raw_dir.glob("*.csv"):
                ds = raw_csv.stem
                metrics_ok = (self.metrics_dir / f"{ds}_{metrics_name}.csv").exists()
                hybrid_ok = (self.forecast_hybrid / f"{ds}_{hybrid_name}.csv").exists()

                if metrics_ok and hybrid_ok:
                    datasets_to_run.append(ds)

            if not datasets_to_run:
                print("   ⚠️ Nenhum dataset possui arquivos de métricas e previsões gerados ainda.")
                return
            else:
                print(f"   ✅ Datasets validados e prontos: {', '.join(datasets_to_run)}")
        else:
            datasets_to_run = [target_dataset]

        for ds_name in datasets_to_run:
            print(f"\n🖼️  Processando imagens para: {ds_name}")
            try:
                if rolling:
                    self.plot_mse_benchmark_rolling(ds_name)
                    self.plot_time_series_forecast_rolling(ds_name)
                else:
                    self.plot_mse_benchmark(ds_name)
                    self.plot_time_series_forecast(ds_name)
            except Exception as e:
                print(f"   ❌ Erro ao gerar gráficos para {ds_name}: {e}")

        print("-" * 60 + "\n")


if __name__ == "__main__":
    plotter = ResultsPlotter()
    plotter.run()
