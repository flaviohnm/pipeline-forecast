import sys
import urllib.request
from pathlib import Path

import pandas as pd

# Define a raiz do projeto e ajusta o sys.path para importar a pasta src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução
from src.utils.general import get_datasets

# Garante que a pasta data/ exista
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def baixar_dataset(dataset_name, info):
    print(f"\n📥 Iniciando ingestão para: {dataset_name}")

    url = info.get("url")
    if not url:
        print(f"⚠️ Nenhuma URL configurada para {dataset_name}. Pulando...")
        return

    # Utiliza a chave "path" definida no general.py
    arquivo_destino = BASE_DIR / info["path"]

    if arquivo_destino.exists():
        print(f"✅ O arquivo {dataset_name}.csv já existe na pasta {arquivo_destino.parent.name}/. Pulando download.")
        return

    print(f"⏳ Baixando {dataset_name} a partir da URL...")
    try:
        # Usa um header de navegador para evitar bloqueios de bots
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            conteudo = response.read()

        with open(arquivo_destino, "wb") as f:
            f.write(conteudo)

        print(f"✅ Download concluído: {info['path']}")

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
    print("\n📥 [PASSO 0] Ingestão de Dados (Lendo Configuração de general.py)")
    print("-" * 60)

    # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO (Substitui o .env)
    datasets_to_run = get_datasets()

    if not datasets_to_run:
        print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando ingestão.")
        exit(1)

    # 2. ITERA SOBRE OS DATASETS CONFIGURADOS NA LISTA
    for dataset_name, info in datasets_to_run.items():
        baixar_dataset(dataset_name, info)

    print("-" * 60)
    print("🏁 Ingestão finalizada.")
