import os
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

# Define caminhos base a partir da localização do script (src/ingestion)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "main_config.yaml"
DATA_DIR = BASE_DIR / "data"

# Garante que a pasta data/ exista
DATA_DIR.mkdir(parents=True, exist_ok=True)

def baixar_dataset(dataset_name, info):
    print(f"\n📥 Iniciando ingestão para: {dataset_name}")
    url = info.get("url")
    if not url:
        print(f"⚠️ Nenhuma URL configurada para {dataset_name} no YAML. Pulando...")
        return

    arquivo_destino = DATA_DIR / f"{dataset_name}.csv"

    if arquivo_destino.exists():
        print(f"✅ O arquivo {dataset_name}.csv já existe na pasta data/. Pulando download.")
        return

    print(f"⏳ Baixando {dataset_name} a partir da URL...")
    try:
        # Usa um header de navegador para evitar bloqueios de bots
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            conteudo = response.read()
        
        with open(arquivo_destino, 'wb') as f:
            f.write(conteudo)
            
        print(f"✅ Download concluído: data/{dataset_name}.csv")

        # Filtragem de colunas
        print(f"🧹 Filtrando colunas essenciais para o dataset {dataset_name}...")
        df = pd.read_csv(arquivo_destino)
        colunas_necessarias = [info["date_column"], info["target_column"]]
        
        if all(col in df.columns for col in colunas_necessarias):
            df = df[colunas_necessarias]
            df.to_csv(arquivo_destino, index=False)
            print(f"✅ {dataset_name} limpo e salvo com as colunas: {colunas_necessarias}.")
        else:
            print(f"⚠️ Aviso: As colunas {colunas_necessarias} não foram encontradas. Arquivo salvo sem alterações.")

    except Exception as e:
        print(f"❌ Erro ao baixar ou processar {dataset_name}: {e}")

if __name__ == "__main__":
    print("\n📥 [PASSO 0] Ingestão de Dados (Lendo Configurações do YAML)")
    print("-" * 60)

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Arquivo de configuração não encontrado em: {CONFIG_PATH}")
        exit(1)

    # Lê a variável de ambiente (padrão é ETTh1 se não encontrar o .env)
    dataset_env = os.getenv("DATASET", "ETTh1").strip()

    datasets_config = config.get("datasets", {})

    if dataset_env.upper() == "ALL":
        print("🚀 Modo BATCH detectado: Baixando TODOS os datasets configurados.")
        for dataset_name, info in datasets_config.items():
            baixar_dataset(dataset_name, info)
    else:
        # Modo Dataset Único
        if dataset_env in datasets_config:
            baixar_dataset(dataset_env, datasets_config[dataset_env])
        else:
            print(f"❌ Dataset '{dataset_env}' não encontrado no arquivo main_config.yaml.")

    print("-" * 60)
    print("🏁 Ingestão finalizada.")