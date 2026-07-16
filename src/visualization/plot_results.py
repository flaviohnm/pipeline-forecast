import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as plt_sns
import yaml

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class ResultsPlotter:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.metrics_dir = BASE_DIR / self.config["results_paths"]["metrics"]
        self.plots_dir = BASE_DIR / self.config["results_paths"]["plots"]
        self.forecast_hybrid = BASE_DIR / self.config["results_paths"]["forecasts"] / "hybrid"

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
        info = self.config["datasets"][dataset_name]
        raw_file = BASE_DIR / info["path"]

        if not preds_file.exists() or not raw_file.exists():
            print("   ⚠️ Arquivos de previsão ou dados reais não encontrados.")
            return

        # Carrega dados reais e previsões do Híbrido e ARIMA
        df_preds = pd.read_csv(preds_file)
        df_preds["ds"] = pd.to_datetime(df_preds["ds"])

        df_raw = pd.read_csv(raw_file)
        df_raw = df_raw[[info["date_column"], info["target_column"]]].copy()
        df_raw.columns = ["ds", "Real"]
        df_raw["ds"] = pd.to_datetime(df_raw["ds"])

        # Junta os dados
        df_plot = pd.merge(df_preds, df_raw, on="ds", how="inner")

        # Vamos focar apenas nos últimos 192 pontos para o gráfico não ficar confuso
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
            f"Previsão de Curto/Médio Prazo: Real vs Modelos ({dataset_name} - Últimas {zoom_horizon}h)",
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

    def run(self):
        print("\n🎨 [PASSO 5] Gerando Visualizações para o Artigo Científico")
        
        # Puxa a variável de ambiente (ou ETTh1 por padrão se estiver vazio)
        target_dataset = os.getenv("DATASET", "ETTh1").strip()

        if target_dataset.upper() == "ALL":
            print("🚀 Modo BATCH detectado: Filtrando datasets com dados PRONTOS para plotagem...")
            datasets_to_run = []
            
            # Só adiciona na fila se os arquivos já existirem nas pastas
            for ds in self.config["datasets"].keys():
                metrics_ok = (self.metrics_dir / f"{ds}_evaluation_complete.csv").exists()
                hybrid_ok = (self.forecast_hybrid / f"{ds}_hybrid_predictions.csv").exists()
                
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
                self.plot_mse_benchmark(ds_name)
                self.plot_time_series_forecast(ds_name)
            except Exception as e:
                print(f"   ❌ Erro ao gerar gráficos para {ds_name}: {e}")
                
        print("-" * 60 + "\n")


if __name__ == "__main__":
    plotter = ResultsPlotter()
    plotter.run()