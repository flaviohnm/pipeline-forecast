import numpy as np
import pandas as pd
import pytest

from src.ingestion.preflight import check_dataset
from src.metrics.check_outputs import missing_for_dataset
from src.metrics.evaluate_metrics import MODEL_COLUMNS
from src.metrics.statistical_tests import build_friedman_blocks
from src.utils.general import DATASETS_INFO, get_datasets, series_problems

INFO = {"date_column": "date", "target_column": "OT", "freq": "h", "missing_values": [-9999]}


def _serie(n=50, **kw):
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="h"), "OT": np.arange(n, dtype=float)})


def test_series_problems_limpa_e_suja():
    assert series_problems(_serie(), INFO) == []
    suja = _serie()
    suja.loc[3, "OT"] = -9999
    suja = suja.drop(index=10)
    texto = " | ".join(series_problems(suja, INFO))
    assert "grade regular" in texto and "sentinela" in texto
    dup = pd.concat([_serie(), _serie().iloc[[5]]]).sort_values("date", kind="stable")
    assert any("repetida" in p for p in series_problems(dup, INFO))


def test_pf_datasets_tem_prioridade_sobre_execution_list(monkeypatch):
    monkeypatch.setenv("PF_DATASETS", "Weather, ECL")
    assert list(get_datasets("rolling")) == ["Weather", "ECL"]
    monkeypatch.setenv("PF_DATASETS", "NaoExiste")
    assert get_datasets("rolling") == {}
    monkeypatch.delenv("PF_DATASETS")
    assert set(get_datasets("rolling")) <= set(DATASETS_INFO)


def test_preflight_detecta_bronze_ausente_e_irregular(tmp_path):
    info = {**DATASETS_INFO["ETTh1"], "path": "data/bronze/ETTh1.csv"}
    assert "bronze não encontrado" in check_dataset("ETTh1", info, base_dir=tmp_path)[0]
    (tmp_path / "data/bronze").mkdir(parents=True)
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=14400, freq="h"), "OT": 1.0})
    df.drop(index=100).to_csv(tmp_path / "data/bronze/ETTh1.csv", index=False)
    problemas = check_dataset("ETTh1", {**info, "date_column": "date", "target_column": "OT"}, base_dir=tmp_path)
    assert any("grade regular" in p for p in problemas)
    df.to_csv(tmp_path / "data/bronze/ETTh1.csv", index=False)
    assert check_dataset("ETTh1", {**info, "date_column": "date", "target_column": "OT"}, base_dir=tmp_path) == []


def _avaliacao(modelos, horizontes, n_janelas):
    return pd.DataFrame(
        [{"Horizonte": h, "Modelo": m, "n_janelas": n_janelas} for h in horizontes for m in modelos]
    )


def test_conferencia_final_acusa_modelo_e_janelas_faltando(tmp_path):
    from src.utils.general import get_rolling_origins

    info = DATASETS_INFO["ETTh1"]
    n_rows = 14400
    n = len(get_rolling_origins(info, n_rows))
    horizontes = info["forecast_horizon"]
    _avaliacao(list(MODEL_COLUMNS), horizontes, n).to_csv(tmp_path / "ETTh1_evaluation_rolling.csv", index=False)
    for extra in ("residual_check", "hybrid_alpha"):
        (tmp_path / f"ETTh1_{extra}.csv").write_text("x\n1\n")
    assert missing_for_dataset("ETTh1", info, tmp_path, n_rows) == []

    modelos = [m for m in MODEL_COLUMNS if "ARIMA + N-HiTS" not in m]
    _avaliacao(modelos, horizontes, n - 1).to_csv(tmp_path / "ETTh1_evaluation_rolling.csv", index=False)
    (tmp_path / "ETTh1_hybrid_alpha.csv").unlink()
    texto = " | ".join(missing_for_dataset("ETTh1", info, tmp_path, n_rows))
    assert "sem Híbrido (ARIMA + N-HiTS)" in texto and "janelas incompletas" in texto and "hybrid_alpha" in texto


def test_friedman_ignora_modelos_ausentes_em_algum_dataset():
    rows = []
    for ds, modelos in {"A": ["m1", "m2", "m3", "m4"], "B": ["m1", "m2", "m3"]}.items():
        for m in modelos:
            rows.append({"Dataset": ds, "Horizonte": 24, "Modelo": m, "MSE": 1.0 + len(m) * 0 + hash(m) % 5})
    blocks, desc = build_friedman_blocks(pd.DataFrame(rows))
    assert desc == "dataset" and list(blocks[24].columns) == ["m1", "m2", "m3"]
