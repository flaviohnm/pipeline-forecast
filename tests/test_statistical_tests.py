import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.metrics.statistical_tests import (
    build_friedman_blocks,
    diebold_mariano,
    dm_table,
    friedman_iman_davenport,
    friedman_tables,
    holm_adjust,
    max_lag_for_overlap,
    nemenyi_critical_difference,
    posthoc_pairwise,
    run_dm_for_dataset,
    run_friedman,
)

# --- holm_adjust -------------------------------------------------------------


def test_holm_adjust_known_values():
    # p ordenados 0.01, 0.02, 0.03 (m=3): 0.03, max(0.03, 0.04)=0.04, max(0.04, 0.03)=0.04
    assert holm_adjust([0.03, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])


def test_holm_adjust_caps_at_one_and_never_below_raw():
    p = np.array([0.5, 0.9, 0.04])
    adj = holm_adjust(p)
    assert (adj <= 1).all()
    assert (adj >= p - 1e-12).all()


# --- max_lag_for_overlap ----------------------------------------------------


@pytest.mark.parametrize(
    "h,step,expected",
    [(24, 168, 0), (168, 168, 0), (169, 168, 1), (336, 168, 1), (720, 168, 4)],
)
def test_max_lag_for_overlap(h, step, expected):
    assert max_lag_for_overlap(h, step) == expected


def test_max_lag_invalid_step():
    with pytest.raises(ValueError):
        max_lag_for_overlap(24, 0)


# --- diebold_mariano ---------------------------------------------------------


def test_dm_detects_clear_difference():
    rng = np.random.default_rng(0)
    b = rng.uniform(0.9, 1.1, 40)
    a = b + 0.5 + rng.normal(0, 0.05, 40)  # A sempre pior
    res = diebold_mariano(a, b)
    assert res["diferenca_media"] > 0
    assert res["p_valor"] < 0.001


def test_dm_no_difference_gives_large_p():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.9, 1.1, 60)
    b = rng.uniform(0.9, 1.1, 60)
    assert diebold_mariano(a, b)["p_valor"] > 0.05


def test_dm_matches_hln_formula_without_lags():
    rng = np.random.default_rng(2)
    a, b = rng.normal(1, 0.3, 30), rng.normal(1.1, 0.3, 30)
    d = a - b
    n = len(d)
    dm = d.mean() / np.sqrt(d.var(ddof=0) / n)
    expected = dm * np.sqrt((n - 1) / n)  # h=1 => (n+1-2+0)/n
    assert diebold_mariano(a, b, max_lag=0)["estatistica"] == pytest.approx(expected)


def test_dm_is_antisymmetric():
    rng = np.random.default_rng(3)
    a, b = rng.normal(1, 0.3, 25), rng.normal(1.2, 0.3, 25)
    r1, r2 = diebold_mariano(a, b, 2), diebold_mariano(b, a, 2)
    assert r1["estatistica"] == pytest.approx(-r2["estatistica"])
    assert r1["p_valor"] == pytest.approx(r2["p_valor"])


def test_dm_identical_or_too_short_returns_nan():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(diebold_mariano(x, x)["p_valor"])
    assert np.isnan(diebold_mariano([1.0, 2.0], [2.0, 1.0])["p_valor"])


def test_dm_more_lags_widen_uncertainty_for_autocorrelated_differentials():
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 1, 60)
    d = np.convolve(noise, np.ones(5) / 5, mode="same") + 0.15  # diferencial autocorrelacionado
    p0 = diebold_mariano(d, np.zeros_like(d), max_lag=0)["p_valor"]
    p4 = diebold_mariano(d, np.zeros_like(d), max_lag=4)["p_valor"]
    assert p4 > p0


def test_dm_shape_mismatch():
    with pytest.raises(ValueError):
        diebold_mariano([1, 2, 3], [1, 2])


# --- Friedman / Iman-Davenport ----------------------------------------------


def test_friedman_matches_scipy_chi2():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(15, 4)) + np.array([0, 0.3, 0.6, 0.9])
    res = friedman_iman_davenport(x)
    chi2, p = stats.friedmanchisquare(*[x[:, j] for j in range(4)])
    assert res["Q"] == pytest.approx(chi2)
    assert res["p_chi2"] == pytest.approx(p)


def test_friedman_iman_davenport_formula():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(12, 5)) + np.arange(5) * 0.5
    r = friedman_iman_davenport(x)
    n, k, q = r["N"], r["k"], r["Q"]
    assert r["F_ID"] == pytest.approx((n - 1) * q / (n * (k - 1) - q))
    assert r["p_ID"] == pytest.approx(stats.f.sf(r["F_ID"], k - 1, (k - 1) * (n - 1)))


def test_friedman_perfect_agreement():
    x = np.tile(np.array([1.0, 2.0, 3.0]), (8, 1))
    r = friedman_iman_davenport(x)
    assert list(r["ranks_medios"]) == [1.0, 2.0, 3.0]
    assert r["p_ID"] == 0.0


def test_friedman_ties_get_average_rank():
    x = np.array([[1.0, 1.0, 2.0]] * 5)
    assert list(friedman_iman_davenport(x)["ranks_medios"]) == [1.5, 1.5, 3.0]


def test_friedman_rejects_nan_and_tiny_input():
    with pytest.raises(ValueError):
        friedman_iman_davenport(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        friedman_iman_davenport(np.array([[1.0, 2.0]]))


def test_nemenyi_cd_matches_demsar_table():
    # Demšar (2006), Tabela 5: q_0.05 para k=4 é 2.569 => CD = 2.569*sqrt(4*5/(6*N))
    n = 14
    expected = 2.569 * np.sqrt(4 * 5 / (6 * n))
    assert nemenyi_critical_difference(4, n) == pytest.approx(expected, rel=2e-3)


def test_posthoc_pairs_and_significance():
    ph = posthoc_pairwise(["A", "B", "C"], np.array([1.0, 2.0, 3.0]), n=30)
    assert len(ph) == 3
    row = ph[(ph["Modelo_A"] == "A") & (ph["Modelo_B"] == "C")].iloc[0]
    assert row["significativo"]
    assert (ph["p_holm"] >= ph["p_valor"] - 1e-12).all()


def test_friedman_tables_drops_incomplete_models():
    rng = np.random.default_rng(7)
    wide = pd.DataFrame(rng.normal(size=(10, 3)) + [0, 0.5, 1.0], columns=["A", "B", "C"])
    wide.loc[0, "C"] = np.nan
    summary, ranks, posthoc = friedman_tables({24: wide})
    assert summary.loc[0, "k_modelos"] == 2
    assert set(ranks["Modelo"]) == {"A", "B"}
    assert len(posthoc) == 1


# --- dm_table / blocos -------------------------------------------------------


def _windows(n_windows=13, horizons=(24, 720), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    cutoffs = pd.date_range("2018-01-01", periods=n_windows, freq="168h")
    for c in cutoffs:
        for h in horizons:
            base = rng.uniform(0.05, 0.1)
            for model, delta in (("Bom", 0.0), ("Ruim", 0.05), ("Igual", 0.001)):
                loss = base + delta + rng.normal(0, 0.002)
                rows.append({"cutoff": c, "Horizonte": h, "Modelo": model, "MSE": loss, "MAE": loss**0.5})
    return pd.DataFrame(rows)


def test_dm_table_structure_and_lags():
    df = dm_table(_windows(), horizons=[24, 720], step=168)
    assert set(df["Horizonte"]) == {24, 720}
    assert set(df["Metrica"]) == {"MSE", "MAE"}
    assert df.loc[df["Horizonte"] == 24, "lags"].eq(0).all()
    assert df.loc[df["Horizonte"] == 720, "lags"].eq(4).all()
    ruim = df[(df["Modelo_A"] == "Bom") & (df["Modelo_B"] == "Ruim") & (df["Horizonte"] == 24) & (df["Metrica"] == "MSE")]
    assert ruim.iloc[0]["diferenca_media"] < 0
    assert ruim.iloc[0]["significativo"]


def test_dm_table_uses_only_common_windows():
    w = _windows()
    w = w.drop(w[(w["Modelo"] == "Igual") & (w["cutoff"] == w["cutoff"].min())].index)
    df = dm_table(w, horizons=[24], step=168)
    pair_with_igual = df[(df["Modelo_B"] == "Igual") & (df["Metrica"] == "MSE")]
    assert (pair_with_igual["n"] == 12).all()


def test_build_friedman_blocks_single_dataset_is_marked_exploratory():
    w = _windows()
    w["Dataset"] = "ETTh1"
    blocks, desc = build_friedman_blocks(w)
    assert "exploratório" in desc
    assert blocks[24].shape == (13, 3)


def test_build_friedman_blocks_multiple_datasets_use_dataset_as_block():
    w1, w2, w3 = _windows(seed=1), _windows(seed=2), _windows(seed=3)
    for w, name in ((w1, "ETTh1"), (w2, "ETTh2"), (w3, "ETTm1")):
        w["Dataset"] = name
    blocks, desc = build_friedman_blocks(pd.concat([w1, w2, w3]))
    assert desc == "dataset"
    assert blocks[24].shape == (3, 3)


# --- run_dm_for_dataset / run_friedman (E/S de CSV) --------------------------

INFO = {"forecast_horizon": [24, 720], "rolling": {"step_size": 168}}


def _write_windows_csv(metrics_dir, dataset, seed):
    # Mesmo formato gravado por evaluate_metrics: métricas arredondadas e coluna Normalizacao
    w = _windows(seed=seed)
    w[["MSE", "MAE"]] = w[["MSE", "MAE"]].round(4)
    w["Normalizacao"] = "train"
    w.to_csv(metrics_dir / f"{dataset}_evaluation_rolling_windows.csv", index=False)


def test_run_dm_for_dataset_saves_csv(tmp_path):
    df = run_dm_for_dataset("ETTh1", INFO, _windows(), tmp_path)
    out = tmp_path / "ETTh1_dm_tests.csv"
    assert out.exists()
    saved = pd.read_csv(out)
    assert len(saved) == len(df) == 2 * 2 * 3  # horizontes x métricas x pares
    assert set(saved["lags"]) == {0, 4}  # passo de 168 h vindo de info["rolling"]
    assert {"Modelo_A", "Modelo_B", "p_valor", "p_holm", "significativo"} <= set(saved.columns)


def test_run_dm_for_dataset_single_model_writes_empty_csv(tmp_path):
    w = _windows()
    df = run_dm_for_dataset("ETTh1", INFO, w[w["Modelo"] == "Bom"], tmp_path)
    assert df.empty
    assert (tmp_path / "ETTh1_dm_tests.csv").exists()


def test_run_dm_for_dataset_requires_rolling_info(tmp_path):
    with pytest.raises(KeyError):
        run_dm_for_dataset("ETTh1", {"forecast_horizon": [24]}, _windows(), tmp_path)


def test_run_friedman_without_files_returns_none(tmp_path):
    assert run_friedman(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_run_friedman_single_dataset_is_exploratory(tmp_path):
    _write_windows_csv(tmp_path, "ETTh1", seed=0)
    summary, ranks, posthoc = run_friedman(tmp_path)
    for name in ("friedman_summary.csv", "friedman_ranks.csv", "friedman_posthoc.csv"):
        assert (tmp_path / name).exists()
    assert summary["Bloco"].str.contains("exploratório").all()
    assert (summary["N_blocos"] == 13).all()
    assert (summary["Metrica"] == "MSE").all()
    bom = ranks[(ranks["Horizonte"] == 24) & (ranks["Modelo"] == "Bom")]["rank_medio"].iloc[0]
    assert bom == ranks[ranks["Horizonte"] == 24]["rank_medio"].min()


def test_run_friedman_multiple_datasets_use_dataset_as_block(tmp_path):
    for seed, name in enumerate(("ETTh1", "ETTh2", "ETTm1")):
        _write_windows_csv(tmp_path, name, seed)
    # O resumo (sem sufixo _windows) não deve ser lido
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "ETTh1_evaluation_rolling.csv", index=False)
    summary, _, posthoc = run_friedman(tmp_path, metric="MAE")
    assert (summary["Bloco"] == "dataset").all()
    assert (summary["N_blocos"] == 3).all()
    assert (summary["Metrica"] == "MAE").all()
    assert set(summary["Horizonte"]) == {24, 720}
    assert len(posthoc) == 2 * 3  # horizontes x pares


def test_run_friedman_insufficient_data_returns_none(tmp_path):
    w = _windows()
    w[w["Modelo"] == "Bom"].to_csv(tmp_path / "ETTh1_evaluation_rolling_windows.csv", index=False)
    assert run_friedman(tmp_path) is None
    assert not (tmp_path / "friedman_summary.csv").exists()
