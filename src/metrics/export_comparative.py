import os
from pathlib import Path

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Força o uso do Weather para o diagnóstico
dataset_name = os.getenv("DATASET", "Weather")

class ComparativeExporter:
    def __init__(self):
        with open(BASE_DIR / "config/main_config.yaml", "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.hybrid_dir = BASE_DIR / self.config["results_paths"]["hybrid"]
        self.deep_dir = BASE_DIR / self.config["results_paths"]["deep_learning"]
        self.info = self.config["datasets"][dataset_name]

    def run(self):
        print(f"\n🔍 Gerando Visão Comparativa (Gabarito) para: {dataset_name}")

        # 1. Carrega os Dados Reais (A Referência)
        df_raw = pd.read_csv(BASE_DIR / self.info["path"])
        df_raw = df_raw[[self.info["date_column"], self.info["target_column"]]].copy()
        df_raw.columns = ["ds", "Valor_Real"]
        df_raw["ds"] = pd.to_datetime(df_raw["ds"])

        # 2. Carrega as Previsões de Deep Learning
        deep_file = self.deep_dir / f"{dataset_name}_deep_predictions.csv"
        df_comp = df_raw.copy()
        
        if deep_file.exists():
            df_deep = pd.read_csv(deep_file)
            df_deep["ds"] = pd.to_datetime(df_deep["ds"])
            df_deep = df_deep.drop(columns=["unique_id"], errors="ignore")
            # Junta com os dados reais mantendo apenas as datas que o modelo previu
            df_comp = pd.merge(df_comp, df_deep, on="ds", how="inner")
            print(f"   ✅ Redes Neurais carregadas. Linhas atuais: {len(df_comp)}")
        else:
            print("   ⚠️ Arquivo de Deep Learning não encontrado.")

        # 3. Carrega Híbrido / ARIMA
        hybrid_file = self.hybrid_dir / f"{dataset_name}_hybrid_predictions.csv"
        if hybrid_file.exists():
            df_hybrid = pd.read_csv(hybrid_file)
            df_hybrid["ds"] = pd.to_datetime(df_hybrid["ds"])
            df_hybrid = df_hybrid.drop(columns=["unique_id"], errors="ignore")
            # Adiciona ao dataframe principal
            df_comp = pd.merge(df_comp, df_hybrid, on="ds", how="left")
            print("   ✅ ARIMA/Híbrido carregado.")
        else:
            print("   ⚠️ Arquivo Híbrido não encontrado.")

        # 4. Salva o arquivo final
        out_file = BASE_DIR / f"results/{dataset_name}_comparative_view.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_comp.to_csv(out_file, index=False)
        
        print(f"\n💾 Arquivo comparativo salvo com sucesso em: {out_file}")
        print("💡 DICA: Abra este CSV no Excel e crie um Gráfico de Linhas com as colunas 'Valor_Real', 'ARIMA' e 'NHITS'!")

if __name__ == "__main__":
    exporter = ComparativeExporter()
    exporter.run()