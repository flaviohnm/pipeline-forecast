import sys
import urllib.request
from pathlib import Path

import pandas as pd

# Define a raiz do projeto e ajusta o sys.path para importar a pasta src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução (já lê data_paths do main_config.yaml)
from src.utils.general import BRONZE_DIR as _BRONZE_DIR
from src.utils.general import RAW_DIR as _RAW_DIR
from src.utils.general import clean_series, get_datasets, series_problems

# Garante que as pastas data/raw/ e data/bronze/ existam (mesma fonte que general.py usa)
RAW_DIR = BASE_DIR / _RAW_DIR
BRONZE_DIR = BASE_DIR / _BRONZE_DIR
RAW_DIR.mkdir(parents=True, exist_ok=True)
BRONZE_DIR.mkdir(parents=True, exist_ok=True)


def bronze_problems(bronze_path, info):
    """Problemas do arquivo bronze (lista vazia = íntegro); arquivo ilegível conta como problema."""
    try:
        return series_problems(pd.read_csv(bronze_path), info)
    except Exception as e:
        return [f"arquivo ilegível ({e})"]


def baixar_dataset(dataset_name, info):
    print(f"\n📥 Iniciando ingestão para: {dataset_name}")

    raw_path = BASE_DIR / info["raw_path"]
    bronze_path = BASE_DIR / info["path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    bronze_path.parent.mkdir(parents=True, exist_ok=True)

    if bronze_path.exists():
        # Um bronze antigo (gerado antes da limpeza) não pode ser reaproveitado às cegas
        problemas = bronze_problems(bronze_path, info)
        if not problemas:
            print(f"✅ O arquivo {dataset_name} já existe em {bronze_path.parent.name}/ (bronze) e está íntegro. Pulando.")
            return True
        print(f"♻️  Bronze de {dataset_name} com problemas ({'; '.join(problemas)}). Regenerando a partir do raw...")

    # 1. RAW: baixa o arquivo exatamente como veio da fonte, sem tratar nada
    if raw_path.exists():
        print(f"♻️  Raw de {dataset_name} já existe em {raw_path.parent.name}/. Reaproveitando (sem novo download).")
    else:
        url = info.get("url")
        if not url:
            print(f"❌ Nenhuma URL configurada para {dataset_name} e sem raw local.")
            return False

        print(f"⏳ Baixando {dataset_name} a partir da URL...")
        try:
            # Usa um header de navegador para evitar bloqueios de bots
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                conteudo = response.read()

            with open(raw_path, "wb") as f:
                f.write(conteudo)

            print(f"✅ Download concluído (raw): {info['raw_path']}")
        except Exception as e:
            print(f"❌ Erro ao baixar {dataset_name}: {e}")
            return False

    # 2. BRONZE: filtra as colunas essenciais a partir do raw
    print(f"🧹 Filtrando colunas essenciais para o dataset {dataset_name}...")
    try:
        df = pd.read_csv(raw_path)
        colunas_necessarias = [info["date_column"], info["target_column"]]

        if all(col in df.columns for col in colunas_necessarias):
            df, resumo = clean_series(df[colunas_necessarias], info)
            if resumo["duplicadas"] or resumo["sentinelas"] or resumo["ausentes_apos_grade"]:
                print(
                    f"   🧽 Limpeza: {resumo['duplicadas']} data(s) repetida(s) removida(s), "
                    f"{resumo['sentinelas']} valor(es) sentinela e {resumo['ausentes_apos_grade']} ponto(s) ausente(s) "
                    f"na grade '{info['freq']}' interpolados ({resumo['linhas_antes']} -> {resumo['linhas_depois']} linhas)."
                )
            df.to_csv(bronze_path, index=False)
            print(f"✅ {dataset_name} limpo e salvo em {info['path']} com as colunas: {colunas_necessarias}.")
        else:
            print(f"⚠️ Aviso: As colunas {colunas_necessarias} não foram encontradas em {dataset_name}.")
            print(f"   Salvando cópia integral do raw em {info['path']} (sem filtragem).")
            df.to_csv(bronze_path, index=False)
    except Exception as e:
        print(f"❌ Erro ao processar (raw -> bronze) {dataset_name}: {e}")
        return False

    problemas = bronze_problems(bronze_path, info)
    if problemas:
        print(f"❌ Bronze de {dataset_name} continua com problemas: {'; '.join(problemas)}")
        return False
    return True


if __name__ == "__main__":
    print("\n📥 [PASSO 0] Ingestão de Dados (Lendo Configuração de general.py)")
    print("-" * 60)

    # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO (Substitui o .env)
    datasets_to_run = get_datasets()

    if not datasets_to_run:
        print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando ingestão.")
        exit(1)

    # 2. ITERA SOBRE OS DATASETS CONFIGURADOS NA LISTA
    falhas = [nome for nome, info in datasets_to_run.items() if not baixar_dataset(nome, info)]

    print("-" * 60)
    if falhas:
        print(f"❌ Ingestão incompleta; datasets com problema: {', '.join(falhas)}")
        sys.exit(1)
    print("🏁 Ingestão finalizada.")
