"""
Verificação prévia (antes de gastar horas de treino): para cada dataset selecionado, confere se o bronze
existe, está íntegro (sem data repetida, buraco, ausente ou sentinela) e comporta o protocolo rolling
(treino/validação/teste e ao menos uma janela de validação do horizonte máximo).

Saída: uma linha "OK <dataset>" ou "ERRO <dataset>: motivo" por dataset; código de saída 1 se algum falhar.
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.utils.general import get_datasets, get_rolling_origins, series_problems


def check_dataset(name, info, base_dir=BASE_DIR):
    """Retorna a lista de problemas do dataset (vazia = pronto para rodar)."""
    path = base_dir / info["path"]
    if not path.exists():
        return [f"bronze não encontrado em {info['path']} (rode a ingestão)"]
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return [f"bronze ilegível: {e}"]
    problemas = series_problems(df, info)
    try:
        get_rolling_origins(info, n_rows=len(df))
    except ValueError as e:
        problemas.append(str(e))
    rolling = info.get("rolling") or {}
    if rolling and rolling.get("val_size", 0) < info["max_horizon"]:
        problemas.append(
            f"validação ({rolling['val_size']}) menor que o horizonte máximo ({info['max_horizon']}): sem alpha"
        )
    return problemas


def main():
    datasets = get_datasets("rolling")
    if not datasets:
        print("ERRO nenhum dataset selecionado")
        return 1
    falhou = False
    for name, info in datasets.items():
        problemas = check_dataset(name, info)
        if problemas:
            falhou = True
            print(f"ERRO {name}: " + "; ".join(problemas))
        else:
            print(f"OK {name}")
    return 1 if falhou else 0


if __name__ == "__main__":
    sys.exit(main())
