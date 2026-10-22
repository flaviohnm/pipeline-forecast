import numpy as np
import pandas as pd
import pytest

from src.metrics.evaluate_metrics import ROLLING_SOURCES, MetricsMultiHorizon, evaluate_rolling
from src.utils.general import DATASETS_INFO, get_rolling_origins

# --- get_rolling_origins ------------------------------------------------------


def test_origins_etth1_protocol():
    origins = get_rolling_origins(DATASETS_INFO["ETTh1"], n_rows=17420)
    assert len(origins) == 13
    assert origins[0] == 8640 + 2880  # primeira previsão = início do bloco de teste
    assert all(b - a == 168 for a, b in zip(origins, origins[1:]))
    assert origins[-1] + 720 <= 8640 + 2880 + 2880  # última janela cabe no bloco de teste


def test_origins_last_window_fits_exactly_when_aligned():
    info = {"max_horizon": 4, "train_size": 10, "rolling": {"val_size": 2, "test_size": 8, "step_size": 2}}
    assert get_rolling_origins(info, n_rows=20) == [12, 14, 16]


@pytest.mark.parametrize(
    "info, n_rows, msg",
    [
        ({"max_horizon": 4, "train_size": 10}, 30, "rolling"),
        ({"max_horizon": 4, "rolling": {"val_size": 2, "test_size": 8, "step_size": 1}}, 30, "train_size"),
        ({"max_horizon": 9, "train_size": 10, "rolling": {"val_size": 2, "test_size": 8, "step_size": 1}}, 30, "menor"),
        ({"max_horizon": 4, "train_size": 10, "rolling": {"val_size": 2, "test_size": 8, "step_size": 1}}, 19, "excedem"),
    ],
)
def test_origins_invalid_config(info, n_rows, msg):
    with pytest.raises(ValueError, match=msg):
        get_rolling_origins(info, n_rows)


# --- evaluate_rolling ---------------------------------------------------------


def _windows(n_windows=3, h=4):
    """y_real = 0; o modelo A erra exatamente k na janela k; o B não tem previsão na janela 1."""
    frames = []
    for k in range(n_windows):
        cutoff = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=10 * k)
        ds = cutoff + pd.to_timedelta(np.arange(1, h + 1), unit="h")
        frames.append(
            pd.DataFrame(
                {"cutoff": cutoff, "ds": ds, "y_real": 0.0, "A": float(k), "B": np.nan if k == 1 else 1.0}
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_evaluate_rolling_mean_std_across_windows():
    df_windows, df_summary = evaluate_rolling(_windows(), [2, 4], {"A": "A"}, norm_mean=0.0, norm_std=1.0)
    assert len(df_windows) == 3 * 2
    row = df_summary.query("Horizonte == 4 and Modelo == 'A'").iloc[0]
    assert row.MSE_mean == pytest.approx(np.mean([0, 1, 4]))
    assert row.MSE_std == pytest.approx(np.std([0, 1, 4], ddof=1))
    assert row.MAE_mean == pytest.approx(1.0)
    assert row.n_janelas == 3


def test_evaluate_rolling_applies_normalization():
    _, df_summary = evaluate_rolling(_windows(), [4], {"A": "A"}, norm_mean=5.0, norm_std=2.0)
    assert df_summary.iloc[0].MAE_mean == pytest.approx(1.0 / 2.0)


def test_evaluate_rolling_skips_incomplete_window_only():
    df_windows, df_summary = evaluate_rolling(_windows(), [4], {"A": "A", "B": "B"}, 0.0, 1.0)
    assert df_summary.set_index("Modelo").loc["B", "n_janelas"] == 2
    assert df_summary.set_index("Modelo").loc["A", "n_janelas"] == 3
    assert df_windows.query("Modelo == 'B'")["cutoff"].nunique() == 2


def test_evaluate_rolling_raises_on_short_window():
    df = _windows().iloc[:-1]  # última janela com 3 passos em vez de 4
    with pytest.raises(ValueError, match="Horizonte 4"):
        evaluate_rolling(df, [4], {"A": "A"}, 0.0, 1.0)


# --- process_dataset_rolling (integração com arquivos sintéticos) ------------


@pytest.fixture
def rolling_setup(tmp_path):
    """Série horária puramente diária (período 24) + previsões rolling sintéticas em pastas temporárias."""
    n = 24 * 20
    ds = pd.date_range("2021-01-01", periods=n, freq="h")
    y = 10 + 3 * np.sin(2 * np.pi * np.arange(n) / 24)
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame({"date": ds, "OT": y}).to_csv(raw_path, index=False)

    info = {
        "path": str(raw_path),
        "date_column": "date",
        "target_column": "OT",
        "forecast_horizon": [24, 48],
        "max_horizon": 48,
        "seasonal_period": 24,
        "train_size": 240,
        "rolling": {"val_size": 48, "test_size": 168, "step_size": 24},
    }
    origins = get_rolling_origins(info, n)
    actual = pd.DataFrame({"ds": ds, "y": y})

    def rolling_frame(cols):
        frames = []
        for start in origins:
            w = actual.iloc[start : start + 48].copy()
            w.insert(0, "cutoff", ds[start - 1])
            for name, fn in cols.items():
                w[name] = fn(w["y"].to_numpy())
            frames.append(w)
        return pd.concat(frames, ignore_index=True).assign(unique_id="SYN")

    results_paths = {k: str(tmp_path / k) + "/" for k in ROLLING_SOURCES}
    for k in ROLLING_SOURCES:
        (tmp_path / k).mkdir()
    # ARIMA perfeito; o híbrido repete a coluna ARIMA (não pode duplicar) e soma +1
    rolling_frame({"ARIMA": lambda y: y}).to_csv(tmp_path / "arima" / "SYN_arima_rolling.csv", index=False)
    rolling_frame({"ARIMA": lambda y: y + 99, "Hybrid_ARIMA_NHITS": lambda y: y + 1}).to_csv(
        tmp_path / "hybrid" / "SYN_hybrid_rolling.csv", index=False
    )
    # Deep com nome original do NeuralForecast (NHITS) e sem a última janela
    rolling_frame({"NHITS": lambda y: y - 2}).query("cutoff != cutoff.max()").to_csv(
        tmp_path / "deep_learning" / "SYN_deep_rolling.csv", index=False
    )

    evaluator = MetricsMultiHorizon.__new__(MetricsMultiHorizon)
    evaluator.config = {"results_paths": results_paths}
    evaluator.metrics_dir = tmp_path / "metrics"
    evaluator.metrics_dir.mkdir()
    evaluator.norm_stats = "train"
    evaluator.protocol = "rolling"
    return evaluator, info, origins, tmp_path


def test_process_dataset_rolling_end_to_end(rolling_setup, capsys):
    evaluator, info, origins, tmp_path = rolling_setup
    evaluator.process_dataset_rolling("SYN", info)

    summary = pd.read_csv(tmp_path / "metrics" / "SYN_evaluation_rolling.csv")
    windows = pd.read_csv(tmp_path / "metrics" / "SYN_evaluation_rolling_windows.csv")
    assert list(summary.columns) == [
        "Horizonte", "Modelo", "MSE_mean", "MSE_std", "MAE_mean", "MAE_std", "n_janelas", "Normalizacao"
    ]
    assert list(windows.columns) == ["cutoff", "Horizonte", "Modelo", "MSE", "MAE", "Normalizacao"]
    assert (summary["Normalizacao"] == "train").all()

    s = summary.set_index(["Horizonte", "Modelo"])
    std_train = pd.read_csv(info["path"])["OT"].iloc[:240].std()
    # ARIMA vem do próprio arquivo (perfeito), não da cópia +99 do híbrido
    assert s.loc[(48, "ARIMA"), "MSE_mean"] == 0
    assert s.loc[(48, "Híbrido (ARIMA + N-HiTS)"), "MAE_mean"] == pytest.approx(1 / std_train, abs=1e-4)
    # Série puramente diária: o ingênuo sazonal acerta tudo; o de último valor não
    assert s.loc[(24, "Ingênuo Sazonal"), "MSE_mean"] == 0
    assert s.loc[(24, "Ingênuo (último valor)"), "MSE_mean"] > 0
    # Deep renomeado para N-HiTS, com uma janela a menos e aviso no log
    assert s.loc[(24, "N-HiTS"), "n_janelas"] == len(origins) - 1
    assert s.loc[(24, "ARIMA"), "n_janelas"] == len(origins)
    assert "sem previsão em todas" in capsys.readouterr().out


def test_process_dataset_rolling_naive_only(rolling_setup):
    """Sem nenhum arquivo rolling de modelo, avalia só os ingênuos (fase 1 do protocolo)."""
    evaluator, info, origins, tmp_path = rolling_setup
    for f in tmp_path.glob("*/SYN_*_rolling.csv"):
        f.unlink()
    evaluator.process_dataset_rolling("SYN", info)
    summary = pd.read_csv(tmp_path / "metrics" / "SYN_evaluation_rolling.csv")
    assert set(summary["Modelo"]) == {"Ingênuo (último valor)", "Ingênuo Sazonal"}
    assert (summary["n_janelas"] == len(origins)).all()
