import numpy as np
import pandas as pd
import pytest

from src.metrics.evaluate_metrics import (
    BIAS_ALERT_THRESHOLD,
    MetricsMultiHorizon,
    check_bias,
    check_model_columns,
    check_no_train_overlap,
    compute_norm_stats,
    naive_forecasts,
    slice_horizon,
)


@pytest.fixture
def df_eval():
    """Série sintética: senoide diária + ruído, com um modelo bom e um em escala z-score."""
    rng = np.random.default_rng(0)
    n = 720
    ds = pd.date_range("2020-01-01", periods=n, freq="h")
    y = 10 + 3 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "ds": ds,
            "y_real": y,
            "ARIMA": y + rng.normal(0, 0.5, n),  # sem viés
            "ETS": (y - y.mean()) / y.std(),  # escala errada (normalizada)
        }
    )


# --- slice_horizon -----------------------------------------------------------


def test_slice_horizon_returns_first_h_rows(df_eval):
    df_h = slice_horizon(df_eval, 24)
    assert len(df_h) == 24
    assert df_h["ds"].iloc[0] == df_eval["ds"].iloc[0]


def test_slice_horizon_raises_when_horizon_exceeds_rows(df_eval):
    with pytest.raises(ValueError, match="Horizonte 1000"):
        slice_horizon(df_eval, 1000)


def test_slice_horizon_raises_when_merge_dropped_rows(df_eval):
    # Simula datas faltando após o merge inner
    with pytest.raises(ValueError, match="encontradas 700"):
        slice_horizon(df_eval.iloc[20:], 720)


# --- check_model_columns ------------------------------------------------------


def test_check_model_columns_reports_missing(df_eval, capsys):
    missing = check_model_columns(df_eval, {"ARIMA": "ARIMA", "LSTM": "LSTM"})
    assert missing == ["LSTM (LSTM)"]
    assert "ausentes" in capsys.readouterr().out


def test_check_model_columns_silent_when_all_present(df_eval, capsys):
    assert check_model_columns(df_eval, {"ARIMA": "ARIMA", "ETS": "ETS"}) == []
    assert capsys.readouterr().out == ""


# --- check_bias ---------------------------------------------------------------


def test_check_bias_flags_wrong_scale_only(df_eval):
    alerts = check_bias(df_eval, {"ARIMA": "ARIMA", "ETS": "ETS"}, reference_std=df_eval["y_real"].std())
    assert list(alerts) == ["ETS"]
    assert alerts["ETS"] == pytest.approx(-df_eval["y_real"].mean(), abs=1e-6)


def test_check_bias_threshold_boundary(df_eval):
    std = 2.0
    df = df_eval[["y_real"]].copy()
    df["below"] = df["y_real"] + 0.99 * BIAS_ALERT_THRESHOLD * std
    df["above"] = df["y_real"] + 1.01 * BIAS_ALERT_THRESHOLD * std
    alerts = check_bias(df, {"below": "below", "above": "above"}, reference_std=std)
    assert list(alerts) == ["above"]


def test_check_bias_ignores_missing_columns(df_eval):
    assert check_bias(df_eval, {"LSTM": "LSTM"}, reference_std=1.0) == {}


# --- compute_norm_stats / check_no_train_overlap (passo B) -------------------


@pytest.fixture
def df_raw():
    """Série com mudança de nível: treino em torno de 20, resto em torno de 5."""
    ds = pd.date_range("2020-01-01", periods=100, freq="h")
    y = np.r_[np.full(60, 20.0), np.full(40, 5.0)] + np.tile([-1.0, 1.0], 50)
    return pd.DataFrame({"ds": ds, "y_real": y})


def test_norm_stats_global_uses_whole_series(df_raw):
    mean, std = compute_norm_stats(df_raw, "global")
    assert mean == pytest.approx(df_raw["y_real"].mean())
    assert std == pytest.approx(df_raw["y_real"].std())


def test_norm_stats_train_uses_only_train_block(df_raw):
    mean, std = compute_norm_stats(df_raw, "train", train_size=60)
    assert mean == pytest.approx(20.0)
    assert std == pytest.approx(df_raw["y_real"].iloc[:60].std())


def test_norm_stats_train_respects_time_order(df_raw):
    shuffled = df_raw.sample(frac=1, random_state=0)
    assert compute_norm_stats(shuffled, "train", 60) == pytest.approx(compute_norm_stats(df_raw, "train", 60))


@pytest.mark.parametrize("train_size", [None, 0, 100, 500])
def test_norm_stats_train_invalid_size(df_raw, train_size):
    with pytest.raises(ValueError):
        compute_norm_stats(df_raw, "train", train_size)


def test_norm_stats_invalid_mode(df_raw):
    with pytest.raises(ValueError, match="norm_stats"):
        compute_norm_stats(df_raw, "test")


def test_no_train_overlap_ok(df_raw):
    check_no_train_overlap(df_raw, df_raw.iloc[60:], train_size=60)


def test_no_train_overlap_raises(df_raw):
    with pytest.raises(ValueError, match="antes do fim do treino"):
        check_no_train_overlap(df_raw, df_raw.iloc[50:], train_size=60)


def test_normalization_changes_metric_scale(df_raw):
    """Mesmo erro absoluto, MSE diferente conforme o desvio usado: protege contra troca silenciosa."""
    y_true = df_raw["y_real"].iloc[60:]
    y_pred = y_true + 1.0
    evaluator = MetricsMultiHorizon.__new__(MetricsMultiHorizon)  # sem ler config/pastas
    for mode, size in [("global", None), ("train", 60)]:
        mean, std = compute_norm_stats(df_raw, mode, size)
        mse, mae = evaluator.calculate_metrics(y_true, y_pred, mean, std)
        assert mae == pytest.approx(1.0 / std)
        assert mse == pytest.approx(1.0 / std**2)


# --- naive_forecasts (passo C) ------------------------------------------------


@pytest.fixture
def df_daily():
    """5 dias horários: valor = hora do dia + 100 * dia, para rastrear de onde veio cada previsão."""
    ds = pd.date_range("2020-01-01", periods=120, freq="h")
    y = ds.hour + 100.0 * (ds.day - 1)
    return pd.DataFrame({"ds": ds, "y_real": y})


def test_naive_repeats_last_value(df_daily):
    start = df_daily["ds"].iloc[96]  # início do 5º dia
    out = naive_forecasts(df_daily, start, 24, season_length=24)
    assert len(out) == 24
    assert (out["Naive"] == df_daily["y_real"].iloc[95]).all()  # 23h do 4º dia = 323
    assert out["Naive"].iloc[0] == 323.0


def test_seasonal_naive_repeats_last_day(df_daily):
    start = df_daily["ds"].iloc[96]
    out = naive_forecasts(df_daily, start, 24, season_length=24)
    # Previsão para a hora k do dia 5 = valor da hora k do dia 4
    np.testing.assert_array_equal(out["SeasonalNaive"], 300.0 + np.arange(24))
    np.testing.assert_array_equal(out["ds"], df_daily["ds"].iloc[96:])


def test_seasonal_naive_cycles_beyond_one_season(df_daily):
    start = df_daily["ds"].iloc[48]
    out = naive_forecasts(df_daily, start, 72, season_length=24)
    np.testing.assert_array_equal(out["SeasonalNaive"], np.tile(100.0 + np.arange(24), 3))


def test_naive_uses_no_future_data(df_daily):
    start = df_daily["ds"].iloc[48]
    base = naive_forecasts(df_daily, start, 24, season_length=24)
    tampered = df_daily.copy()
    tampered.loc[48:, "y_real"] = -999.0  # altera tudo a partir da origem
    out = naive_forecasts(tampered, start, 24, season_length=24)
    pd.testing.assert_frame_equal(base, out)


def test_naive_raises_with_short_history(df_daily):
    with pytest.raises(ValueError, match="sazonalidade"):
        naive_forecasts(df_daily, df_daily["ds"].iloc[10], 24, season_length=24)


def test_naive_raises_unknown_origin(df_daily):
    with pytest.raises(ValueError, match="Origem"):
        naive_forecasts(df_daily, pd.Timestamp("1999-01-01"), 24, season_length=24)
