import os
from pathlib import Path

import pandas as pd
import yaml

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent
dataset_name = os.getenv("DATASET", "ETTh1")


class HybridCombiner:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Onde buscar as previsões
        self.resid_dir = BASE_DIR / self.config["results_paths"]["residuals"]
        self.arima_dir = BASE_DIR / self.config["results_paths"]["arima"]
        self.hybrid_dir = BASE_DIR / self.config["results_paths"]["hybrid"]
        self.hybrid_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\n🔗 [PASSO 3] Combinando as Previsões (ARIMA + N-HiTS)")
        print("Equação do Artigo: Y_final = ARIMA (Linear) + N-HiTS (Não-Linear)")

        # 1. Carrega as Previsões do ARIMA (A Base Linear)
        arima_file = self.arima_dir / f"{dataset_name}_arima_predictions.csv"
        df_arima = pd.read_csv(arima_file)
        df_arima["ds"] = pd.to_datetime(df_arima["ds"])

        # 2. Carrega as Previsões do N-HiTS (Os Resíduos Ajustados)
        nhits_file = self.resid_dir / f"{dataset_name}_nhits_residuals.csv"
        df_nhits = pd.read_csv(nhits_file)
        df_nhits["ds"] = pd.to_datetime(df_nhits["ds"])

        # 3. Faz o Merge (Garante que estamos somando as coisas na mesma hora exata)
        df_final = pd.merge(df_arima, df_nhits, on="ds", how="inner")

        # 4. O CÁLCULO HÍBRIDO (A Mágica acontece aqui)
        df_final["Hybrid_ARIMA_NHITS"] = df_final["ARIMA"] + df_final["NHITS"]

        # 5. Salvar o resultado consolidado
        out_file = self.hybrid_dir / f"{dataset_name}_hybrid_predictions.csv"
        df_final.to_csv(out_file, index=False)

        print(f"\n✅ Previsão final do modelo híbrido salva em: {self.hybrid_dir.name}/{out_file.name}")
        print("-" * 50)
        print("Amostra das 5 primeiras previsões combinadas:")
        print(df_final[["ds", "ARIMA", "NHITS", "Hybrid_ARIMA_NHITS"]].head())
        print("-" * 50 + "\n")


if __name__ == "__main__":
    combiner = HybridCombiner()
    combiner.run()
