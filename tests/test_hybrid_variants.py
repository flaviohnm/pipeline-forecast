import numpy as np
import pandas as pd
import pytest

from src.models.hybrid_variants import (
    ALPHA_COL,
    BLOCKS_ALPHA_COL,
    BLOCKS_COL,
    add_alpha_column,
    alpha_by_block,
    alpha_by_horizon,
    block_edges,
    blocks_forecast,
    compose_blocks,
    load_hybrid_config,
    pick_alpha,
    residual_input_sizes,
)

HORIZONS = [2, 4, 6]
CUTOFFS = pd.to_datetime(["2020-01-01 00:00", "2020-01-02 00:00"])


def nhits_frame(value_by_h, horizons=HORIZONS, cutoffs=CUTOFFS):
    rows = []
    for h in horizons:
        for c in cutoffs:
            for i in range(1, h + 1):
                rows.append({"H": h, "cutoff": c, "ds": c + pd.Timedelta(hours=i), "NHITS": value_by_h(h, i)})
    return pd.DataFrame(rows)


def arima_frame(max_h=6, cutoffs=CUTOFFS, base=10.0, y=None):
    rows = []
    for c in cutoffs:
        for i in range(1, max_h + 1):
            rows.append(
                {"unique_id": "s", "cutoff": c, "ds": c + pd.Timedelta(hours=i), "ARIMA": base,
                 "y": base if y is None else y(i)}
            )
    return pd.DataFrame(rows)


def test_block_edges():
    assert block_edges([6, 2, 4]) == {2: (0, 2), 4: (2, 4), 6: (4, 6)}


def test_pick_alpha_recovers_scale_and_ties_go_to_zero():
    corr = np.array([1.0, -2.0, 3.0, 0.5])
    a, m, m0, m1 = pick_alpha(0.4 * corr, corr, np.linspace(0, 1, 21))
    assert a == pytest.approx(0.4) and m == pytest.approx(0)
    assert m0 > m and m1 > m
    # correção sem relação com o erro: alpha 0 (ARIMA puro)
    a0, *_ = pick_alpha(np.zeros(4), corr, np.linspace(0, 1, 21))
    assert a0 == 0.0


def test_alpha_by_horizon_per_model():
    # erro real = 0.5 * correção em H=2 e correção inútil (erro 0) em H=4
    nh = nhits_frame(lambda h, i: 1.0, horizons=[2, 4])
    ar = arima_frame(max_h=4, y=lambda i: 10.0)  # erro 0 em todos os passos
    out = alpha_by_horizon(ar, nh, step=0.25)
    assert set(out["H"]) == {2, 4} and (out["alpha"] == 0.0).all()

    ar2 = arima_frame(max_h=4, y=lambda i: 10.5)  # erro 0.5 -> alpha 0.5 com correção 1
    out2 = alpha_by_horizon(ar2, nh, step=0.25)
    assert (out2["alpha"] == 0.5).all()
    assert (out2["mse_val"] < out2["mse_val_arima"]).all()


def test_alpha_by_block_uses_only_block_steps():
    # H=4: passos 1-2 têm correção útil (erro 1, correção 1), passos 3-4 correção inútil (erro 0, correção 5)
    nh = nhits_frame(lambda h, i: 1.0 if i <= 2 else 5.0, horizons=[2, 4])
    ar = arima_frame(max_h=4, y=lambda i: 11.0 if i <= 2 else 10.0)
    blocks = alpha_by_block(ar, nh, step=0.25).set_index("H")
    assert blocks.loc[2, "alpha"] == 1.0  # bloco 1-2
    assert blocks.loc[4, "alpha"] == 0.0  # bloco 3-4 (passos 3-4 do modelo H=4)
    assert (blocks.loc[4, "bloco_inicio"], blocks.loc[4, "bloco_fim"]) == (3, 4)


def test_validation_misaligned_raises():
    nh = nhits_frame(lambda h, i: 1.0, horizons=[2])
    ar = arima_frame(max_h=1)  # não cobre o 2º passo
    with pytest.raises(ValueError, match="não cobrem"):
        alpha_by_horizon(ar, nh)


def test_compose_blocks_takes_step_from_smallest_covering_model():
    nh = nhits_frame(lambda h, i: float(h))  # valor identifica o modelo de origem
    comp = compose_blocks(nh)
    assert len(comp) == len(CUTOFFS) * 6 and not comp.duplicated(["cutoff", "ds"]).any()
    one = comp[comp["cutoff"] == CUTOFFS[0]].sort_values("step")
    assert one["NHITS_blocks"].tolist() == [2.0, 2.0, 4.0, 4.0, 6.0, 6.0]
    assert one["step"].tolist() == [1, 2, 3, 4, 5, 6]


def test_blocks_forecast_columns_and_alpha():
    nh = nhits_frame(lambda h, i: float(h))
    ar = arima_frame()
    plain = blocks_forecast(ar, nh)
    assert BLOCKS_ALPHA_COL not in plain.columns and "H" not in plain.columns and "H_bloco" in plain.columns
    one = plain[plain["cutoff"] == CUTOFFS[0]].sort_values("step")
    assert one[BLOCKS_COL].tolist() == [12.0, 12.0, 14.0, 14.0, 16.0, 16.0]

    w = blocks_forecast(ar, nh, alpha_blocks={2: 1.0, 4: 0.5, 6: 0.0})
    one = w[w["cutoff"] == CUTOFFS[0]].sort_values("step")
    assert one[BLOCKS_ALPHA_COL].tolist() == [12.0, 12.0, 12.0, 12.0, 10.0, 10.0]


def test_blocks_forecast_arima_gap_raises():
    nh = nhits_frame(lambda h, i: 1.0)
    with pytest.raises(ValueError, match="não cobre"):
        blocks_forecast(arima_frame(max_h=5), nh)


def test_add_alpha_column():
    df = pd.DataFrame({"H": [2, 2, 4], "ARIMA": [10.0, 10.0, 10.0], "NHITS": [2.0, 2.0, 2.0]})
    out = add_alpha_column(df, {2: 0.5, 4: 0.0})
    assert out[ALPHA_COL].tolist() == [11.0, 11.0, 10.0]
    with pytest.raises(ValueError, match="Sem alpha"):
        add_alpha_column(df, {2: 0.5})


def test_hybrid_config_defaults_and_validation():
    cfg = load_hybrid_config({})
    assert cfg["alpha"]["enabled"] and cfg["blocks"]["enabled"] and cfg["alpha"]["grid_step"] == 0.05
    assert residual_input_sizes(cfg, {"input_size": 336}) == [336]
    cfg2 = load_hybrid_config({"hybrid": {"residual_input_size": 96}})
    assert residual_input_sizes(cfg2, {"input_size": 336}) == [96]
    cfg3 = load_hybrid_config({"hybrid": {"residual_input_size_grid": [336, 96]}})
    assert residual_input_sizes(cfg3, {"input_size": 336}) == [96, 336]
    with pytest.raises(ValueError):
        load_hybrid_config({"hybrid": {"alpha": {"grid_step": 0}}})
    with pytest.raises(ValueError):
        load_hybrid_config({"hybrid": {"residual_input_size_grid": []}})
