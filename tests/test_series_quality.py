import numpy as np
import pandas as pd
import pytest

from src.utils.general import DATASETS_INFO, check_regular_series, clean_series, prepare_data

INFO = {"date_column": "date", "target_column": "OT", "freq": "10min", "missing_values": [-9999]}


def _suja():
    """Série de 60 pontos com data repetida, buraco de 9 passos e sentinelas, como o Weather."""
    datas = pd.date_range("2020-01-01", periods=60, freq="10min")
    df = pd.DataFrame({"date": datas, "OT": np.arange(60, dtype=float)})
    df.loc[10:12, "OT"] = -9999
    df = df.drop(index=range(30, 39))  # buraco
    return pd.concat([df, df.iloc[[20]]]).sort_values("date", kind="stable").reset_index(drop=True)


def test_clean_series_resolve_duplicada_sentinela_e_buraco():
    limpo, resumo = clean_series(_suja(), INFO)
    assert resumo["duplicadas"] == 1 and resumo["sentinelas"] == 3 and resumo["ausentes_apos_grade"] == 12
    assert len(limpo) == 60 and limpo["OT"].notna().all() and (limpo["OT"] > -9000).all()
    assert pd.DatetimeIndex(limpo["date"]).equals(pd.date_range("2020-01-01", periods=60, freq="10min"))
    # interpolação no tempo: a série original era uma rampa 0..59
    assert np.allclose(limpo["OT"], np.arange(60), atol=1e-9)


def test_clean_series_nao_altera_serie_ja_limpa():
    datas = pd.date_range("2020-01-01", periods=20, freq="h")
    df = pd.DataFrame({"date": datas, "OT": np.random.default_rng(0).normal(size=20)})
    limpo, resumo = clean_series(df, {**INFO, "freq": "h", "missing_values": []})
    assert resumo["duplicadas"] == resumo["sentinelas"] == resumo["ausentes_apos_grade"] == 0
    assert np.allclose(limpo["OT"], df["OT"])


def _ds(df):
    return df.rename(columns={"date": "ds", "OT": "y"})


def test_check_regular_series_recusa_data_repetida_e_buraco():
    with pytest.raises(ValueError, match="repetida"):
        check_regular_series(_ds(_suja()), "10min", "Teste")
    sem_dup = _suja().drop_duplicates("date")
    with pytest.raises(ValueError, match="grade regular"):
        check_regular_series(_ds(sem_dup), "10min", "Teste")


def test_check_regular_series_aceita_serie_limpa_e_recusa_ausente():
    limpo, _ = clean_series(_suja(), INFO)
    check_regular_series(_ds(limpo), "10min", "Teste")
    limpo.loc[5, "OT"] = np.nan
    with pytest.raises(ValueError, match="ausente"):
        check_regular_series(_ds(limpo), "10min", "Teste")


def test_prepare_data_confere_regularidade():
    info = {**DATASETS_INFO["Weather"]}
    with pytest.raises(ValueError):
        prepare_data(_suja(), info, "Weather")
    limpo, _ = clean_series(_suja(), info)
    assert list(prepare_data(limpo, info, "Weather").columns) == ["ds", "y", "unique_id"]


def test_weather_declara_sentinela():
    assert DATASETS_INFO["Weather"]["missing_values"] == [-9999]
