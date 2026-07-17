import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Injeta a raiz do projeto no path do Python se necessário
sys.path.append(str(BASE_DIR))


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
        # 1. Recupera o dataset ativo do ambiente (.env)
        dataset_env = os.getenv("DATASET", "ETTh1").strip()

        # 2. Configura a lista de datasets para rodar (Batch ou Único)
        if dataset_env.upper() == "ALL":
            print("\n🚀 Modo BATCH detectado: Combinando ARIMA + N-HiTS para TODOS os datasets.")
            datasets_to_run = list(self.config["datasets"].keys())
        else:
            datasets_to_run = [dataset_env]

        print("\n🔗 [PASSO 5] Combinando as Previsões (ARIMA + N-HiTS)")
        print("Equação do Artigo: Y_final = ARIMA (Linear) + N-HiTS (Não-Linear)")

        for name in datasets_to_run:
            if name not in self.config["datasets"]:
                print(f"⚠️ Dataset {name} não encontrado no main_config.yaml. Pulando...")
                continue

            # Caminhos dos arquivos específicos deste dataset
            arima_file = self.arima_dir / f"{name}_arima_predictions.csv"
            nhits_file = self.resid_dir / f"{name}_nhits_residuals.csv"
            out_file = self.hybrid_dir / f"{name}_hybrid_predictions.csv"

            # 3. Validação robusta de caminhos
            if not arima_file.exists():
                print(f"⚠️ Previsões do ARIMA não encontradas para {name} em: {arima_file.name}. Pulando combinação.")
                continue
            if not nhits_file.exists():
                print(f"⚠️ Previsões de resíduos do N-HiTS não encontradas para {name} em: {nhits_file.name}. Pulando combinação.")
                continue

            print(f"\n📊 Processando combinação para: {name}")

            # 4. Carrega as Previsões do ARIMA (A Base Linear)
            df_arima = pd.read_csv(arima_file)
            df_arima["ds"] = pd.to_datetime(df_arima["ds"])

            # 5. Carrega as Previsões do N-HiTS (Os Resíduos Ajustados)
            df_nhits = pd.read_csv(nhits_file)
            df_nhits["ds"] = pd.to_datetime(df_nhits["ds"])

            # 6. Faz o Merge (Garante que estamos somando as coisas na mesma hora exata)
            df_final = pd.merge(df_arima, df_nhits, on="ds", how="inner")

            # 7. O CÁLCULO HÍBRIDO (A Mágica acontece aqui)
            if "ARIMA" in df_final.columns and "NHITS" in df_final.columns:
                df_final["Hybrid_ARIMA_NHITS"] = df_final["ARIMA"] + df_final["NHITS"]
            else:
                # Fallback caso a nomenclatura das colunas no arquivo de resíduos do N-HiTS use outra chave
                col_nhits = [col for col in df_final.columns if col not in ["ds", "ARIMA", "unique_id_x", "unique_id_y"]][0]
                df_final["Hybrid_ARIMA_NHITS"] = df_final["ARIMA"] + df_final[col_nhits]

            # 8. Salvar o resultado consolidado
            df_final.to_csv(out_file, index=False)

            print(f"✅ Previsão final do modelo híbrido salva em: {self.hybrid_dir.name}/{out_file.name}")
            print("-" * 50)
            print("Amostra das 5 primeiras previsões combinadas:")
            print(df_final[["ds", "ARIMA", "NHITS", "Hybrid_ARIMA_NHITS"]].head())
            print("-" * 50)

        print("\n🏁 Processo de Combinação Híbrida Finalizado.\n")


if __name__ == "__main__":
    combiner = HybridCombiner()
    combiner.run()