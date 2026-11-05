"""
Conferência final da execução: para cada dataset selecionado, confirma que TODOS os modelos esperados têm
métricas em TODOS os horizontes, com o número esperado de janelas rolling, e que os diagnósticos do híbrido
(residual_check e hybrid_alpha) existem. Saída: lista do que falta; código de saída 1 se algo faltar.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.metrics.evaluate_metrics import MODEL_COLUMNS
from src.utils.general import get_datasets, get_rolling_origins


def missing_for_dataset(name, info, metrics_dir, n_rows):
    """Lista do que falta para o dataset estar completo (vazia = completo)."""
    faltas = []
    eval_file = metrics_dir / f"{name}_evaluation_rolling.csv"
    if not eval_file.exists():
        return [f"{eval_file.name} ausente"]
    df = pd.read_csv(eval_file)
    esperado_janelas = len(get_rolling_origins(info, n_rows))
    got = {(int(r.Horizonte), r.Modelo): int(r.n_janelas) for r in df.itertuples()}
    for h in info["forecast_horizon"]:
        ausentes = [m for m in MODEL_COLUMNS if (h, m) not in got]
        if ausentes:
            faltas.append(f"H={h}: sem {', '.join(ausentes)}")
        incompletos = [f"{m} ({got[(h, m)]}/{esperado_janelas})" for m in MODEL_COLUMNS
                       if (h, m) in got and got[(h, m)] != esperado_janelas]
        if incompletos:
            faltas.append(f"H={h}: janelas incompletas em {', '.join(incompletos)}")
    for extra in ("residual_check", "hybrid_alpha"):
        if not (metrics_dir / f"{name}_{extra}.csv").exists():
            faltas.append(f"{name}_{extra}.csv ausente")
    return faltas


def main():
    with open(BASE_DIR / "config" / "main_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics_dir = BASE_DIR / config["results_paths"]["metrics"]
    datasets = get_datasets("rolling")
    incompleto = False
    for name, info in datasets.items():
        bronze = BASE_DIR / info["path"]
        if not bronze.exists():
            incompleto = True
            print(f"INCOMPLETO {name}: bronze ausente")
            continue
        faltas = missing_for_dataset(name, info, metrics_dir, n_rows=len(pd.read_csv(bronze)))
        if faltas:
            incompleto = True
            print(f"INCOMPLETO {name}:")
            for f in faltas:
                print(f"   - {f}")
        else:
            print(f"COMPLETO {name}: {len(MODEL_COLUMNS)} modelos x {len(info['forecast_horizon'])} horizontes")
    return 1 if incompleto else 0


if __name__ == "__main__":
    sys.exit(main())
