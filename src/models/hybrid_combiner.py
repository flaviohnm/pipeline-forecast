import sys
from pathlib import Path

import pandas as pd
import yaml

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Injeta a raiz do projeto no path do Python se necessário
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução
from src.utils.general import get_datasets


class HybridCombiner:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos o YAML apenas para as configurações de caminhos das pastas (results_paths)
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Onde buscar as previsões
        self.resid_dir = BASE_DIR / self.config["results_paths"]["residuals"]
        self.arima_dir = BASE_DIR / self.config["results_paths"]["arima"]
        self.hybrid_dir = BASE_DIR / self.config["results_paths"]["hybrid"]
        self.hybrid_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\n🔗 [PASSO 5] Combinando as Previsões (ARIMA + N-HiTS)")
        print("Equação do Artigo: Y_final = ARIMA (Linear) + N-HiTS (Não-Linear)")

        # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO (Substitui o .env)
        datasets_to_run = get_datasets()

        if not datasets_to_run:
            print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando.")
            return

        # 2. ITERAÇÃO LIMPA
        for name, info in datasets_to_run.items():
            print(f"\n=======================================")
            print(f"📊 Processando combinação para: {name}")
            print(f"=======================================")

            # Caminhos dos arquivos específicos deste dataset
            arima_file = self.arima_dir / f"{name}_arima_predictions.csv"
            nhits_file = self.resid_dir / f"{name}_nhits_residuals.csv"
            out_file = self.hybrid_dir / f"{name}_hybrid_predictions.csv"

            # 3. Validação robusta de caminhos
            if not arima_file.exists():
                print(f"⚠️ Previsões do ARIMA não encontradas para {name} em: {arima_file.name}. Pulando combinação.")
                continue
            if not nhits_file.exists():
                print(
                    f"⚠️ Previsões de resíduos do N-HiTS não encontradas para {name} em: {nhits_file.name}. Pulando combinação."
                )
                continue

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
                col_nhits = [
                    col for col in df_final.columns if col not in ["ds", "ARIMA", "unique_id_x", "unique_id_y"]
                ][0]
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
