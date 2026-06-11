import os
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent
dataset_name = os.getenv("DATASET", "ETTh1")


class DataDownloader:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def run(self):
        print("\n📥 [PASSO 0] Ingestão de Dados (Lendo Configurações do YAML)")

        for name, info in self.config["datasets"].items():
            if name != dataset_name:
                continue

            # Caminho onde o arquivo deve ser salvo
            file_path = BASE_DIR / info["path"]

            # Garante que a pasta (ex: data/) exista
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.exists():
                print(f"   ✅ {name} já está disponível em: {info['path']}")
            else:
                # Busca a URL diretamente do arquivo de configuração
                url = info.get("download_url")

                if url:
                    print(f"   ⏳ Arquivo não encontrado. Baixando {name} a partir da URL configurada...")
                    try:
                        urllib.request.urlretrieve(url, file_path)
                        print(f"   ✅ Download concluído com sucesso: {info['path']}")
                    except Exception as e:
                        print(f"   ❌ Erro ao baixar {name}: {e}")
                else:
                    print(f"   ⚠️ Nenhuma 'download_url' configurada para o dataset {name} no YAML.")

            # --- O FILTRO DE INTEGRIDADE (GATEKEEPER) ---
            print(f"   🧹 Filtrando colunas essenciais para o dataset {name}...")
            try:
                # Lê o arquivo e garante a extração apenas do estritamente necessário
                df = pd.read_csv(file_path)
                date_col = info["date_column"]
                target_col = info["target_column"]

                if date_col in df.columns and target_col in df.columns:
                    df_filtered = df[[date_col, target_col]].copy()

                    # Sobrescreve o dataset bruto para economizar memória nos treinamentos
                    df_filtered.to_csv(file_path, index=False)
                    print(f"   ✅ {name} limpo e salvo apenas com as colunas: [{date_col}, {target_col}].")
                else:
                    print(f"   ⚠️ AVISO: Colunas '{date_col}' ou '{target_col}' ausentes no CSV baixado!")
            except Exception as e:
                print(f"   ❌ Erro ao processar as colunas do arquivo: {e}")

        print("-" * 60 + "\n")


if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.run()
