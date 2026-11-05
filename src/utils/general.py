import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Define a raiz do projeto para localizar o main_config.yaml
BASE_DIR = Path(__file__).resolve().parent.parent.parent

with open(BASE_DIR / "config" / "main_config.yaml", "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

# Bases das camadas de dados, centralizadas no main_config.yaml
RAW_DIR = _config["data_paths"]["raw"]
BRONZE_DIR = _config["data_paths"]["bronze"]

# ==============================================================================
# 1. CONTROLE DE EXECUÇÃO
# ==============================================================================
# Para rodar todos os datasets, use: ["ALL"]
# Para rodar datasets específicos, use: ["ETTh1", "ECL", "Weather"]
EXECUTION_LIST = ["ALL"]


# ==============================================================================
# 2. DICIONÁRIO CENTRAL DE DATASETS
# ==============================================================================
# "filename" é combinado com RAW_DIR/BRONZE_DIR (vindos do main_config.yaml)
# em get_datasets() para montar "raw_path" e "path" dinamicamente.
DATASETS_INFO = {
    "ETTh1": {
        "filename": "ETTh1.csv",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "h",
        "forecast_horizon": [24, 48, 168, 336, 720],
        "max_horizon": 720,
        "test_size": 2880,
        # 12 meses de treino (protocolo padrão ETT: 12/4/4 meses)
        "train_size": 8640,
        "val_size": 720,
        "seasonal_period": 24,
        # Protocolo rolling (evaluation.protocol): blocos treino/val/teste de 12/4/4 meses,
        # origens a cada 1 semana dentro do bloco de teste
        "rolling": {"val_size": 2880, "test_size": 2880, "step_size": 168},
    },
    "ETTh2": {
        "filename": "ETTh2.csv",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "h",
        "forecast_horizon": [24, 48, 168, 336, 720],
        "max_horizon": 720,
        "test_size": 2880,
        # 12 meses de treino (protocolo padrão ETT: 12/4/4 meses)
        "train_size": 8640,
        "val_size": 720,
        "seasonal_period": 24,
        # Protocolo rolling (evaluation.protocol): blocos treino/val/teste de 12/4/4 meses,
        # origens a cada 1 semana dentro do bloco de teste
        "rolling": {"val_size": 2880, "test_size": 2880, "step_size": 168},
    },
    "ETTm1": {
        "filename": "ETTm1.csv",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "15min",
        "forecast_horizon": [24, 48, 96, 288, 672],
        "max_horizon": 672,
        "test_size": 2880,
        # 12 meses de treino (protocolo padrão ETT: 12/4/4 meses)
        "train_size": 34560,
        "val_size": 720,
        "seasonal_period": 96,
        # Protocolo rolling: 12/4/4 meses em passos de 15 min, origens a cada 1 semana
        "rolling": {"val_size": 11520, "test_size": 11520, "step_size": 672},
    },
    "ETTm2": {
        "filename": "ETTm2.csv",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "15min",
        "forecast_horizon": [24, 48, 96, 288, 672],
        "max_horizon": 672,
        "test_size": 2880,
        # 12 meses de treino (protocolo padrão ETT: 12/4/4 meses)
        "train_size": 34560,
        "val_size": 720,
        "seasonal_period": 96,
        # Protocolo rolling: 12/4/4 meses em passos de 15 min, origens a cada 1 semana
        "rolling": {"val_size": 11520, "test_size": 11520, "step_size": 672},
    },
    "ECL": {
        "filename": "ECL.csv",
        "url": "https://huggingface.co/datasets/Time-HD-Anonymous/High_Dimensional_Time_Series/resolve/main/electricity.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "h",
        "forecast_horizon": [24, 48, 168, 336, 720, 960],
        "max_horizon": 960,
        "test_size": 2880,
        "val_size": 2160,
        "seasonal_period": 24,
        # Protocolo rolling: divisão 7/1/2 do ECL (26.304 h), origens a cada 1 semana.
        # Ajuste os tamanhos se o arquivo baixado tiver outro comprimento (get_rolling_origins confere).
        "train_size": 18412,
        "rolling": {"val_size": 2632, "test_size": 5260, "step_size": 168},
    },
    "Weather": {
        "filename": "Weather.csv",
        "url": "https://huggingface.co/datasets/Time-HD-Anonymous/High_Dimensional_Time_Series/resolve/main/weather.csv",
        "target_column": "OT",
        "date_column": "date",
        "freq": "10min",
        # O arquivo usa -9999 como "sem medição" (50 valores no treino); sem tratar, o desvio-padrão do treino
        # passa de ~18 para ~384 e o ARIMA/as redes ajustam sobre outliers.
        "missing_values": [-9999],
        "forecast_horizon": [24, 48, 168, 336, 720],
        "max_horizon": 720,
        "test_size": 10539,
        "val_size": 5269,
        "seasonal_period": 144,
        # Protocolo rolling: divisão 7/1/2 do Weather (52.696 passos de 10 min), origens a cada 1 semana.
        # Ajuste os tamanhos se o arquivo baixado tiver outro comprimento (get_rolling_origins confere).
        "train_size": 36887,
        "rolling": {"val_size": 5270, "test_size": 10539, "step_size": 1008},
    },
}


# ==============================================================================
# 3. FUNÇÕES UTILITÁRIAS
# ==============================================================================
def _with_computed_paths(info):
    """Combina RAW_DIR/BRONZE_DIR (main_config.yaml) com o filename do dataset."""
    enriched = dict(info)
    enriched["raw_path"] = f"{RAW_DIR}{info['filename']}"
    enriched["path"] = f"{BRONZE_DIR}{info['filename']}"
    return enriched


def has_rolling_config(info):
    """True se o dataset tem 'train_size' e 'rolling' (exigidos pelo protocolo rolling)."""
    return bool(info.get("rolling")) and bool(info.get("train_size"))


def requested_datasets():
    """
    Datasets pedidos: a variável de ambiente PF_DATASETS (nomes separados por vírgula, ex.: "Weather" ou
    "ETTh1,ECL") tem prioridade sobre a EXECUTION_LIST. O orquestrador usa isso para rodar cada dataset
    em um processo próprio, de modo que a falha de um não derruba os outros.
    """
    env = os.environ.get("PF_DATASETS", "").strip()
    if env:
        return [d.strip() for d in env.split(",") if d.strip()]
    return list(EXECUTION_LIST)


def get_datasets(protocol=None):
    """
    Retorna os dicionários correspondentes aos datasets definidos na EXECUTION_LIST (ou PF_DATASETS),
    já com "raw_path" e "path" calculados a partir do main_config.yaml.
    Com protocol="rolling", datasets sem 'train_size'/'rolling' são pulados com aviso
    (em vez de interromper o script e impedir os datasets seguintes).
    """
    # Se "ALL" estiver na lista (ignorando maiúsculas/minúsculas), retorna todos
    requested = requested_datasets()
    if "ALL" in [ds.upper() for ds in requested]:
        selected_datasets = {name: _with_computed_paths(info) for name, info in DATASETS_INFO.items()}
    else:
        selected_datasets = {}
        for ds in requested:
            if ds in DATASETS_INFO:
                selected_datasets[ds] = _with_computed_paths(DATASETS_INFO[ds])
            else:
                print(f"⚠️ Aviso: Dataset '{ds}' não configurado em DATASETS_INFO. Ignorando...")

    if protocol == "rolling":
        for name in [n for n, info in selected_datasets.items() if not has_rolling_config(info)]:
            print(f"⚠️ Aviso: Dataset '{name}' sem 'train_size'/'rolling' em DATASETS_INFO; pulado no protocolo rolling.")
            del selected_datasets[name]

    return selected_datasets


def get_rolling_origins(info, n_rows):
    """
    Posições (na série em ordem temporal) do primeiro passo previsto em cada janela do protocolo rolling.

    Blocos: treino [0, train_size) | validação [train_size, +val_size) | teste [+val_size, +test_size).
    Cada janela prevê max_horizon passos e precisa caber inteira no bloco de teste; as origens
    avançam step_size passos. O modelo usa dados até a posição (origem - 1), chamada de "cutoff".
    """
    if not has_rolling_config(info):
        raise ValueError("Protocolo rolling exige 'train_size' e 'rolling' configurados para o dataset.")

    rolling = info["rolling"]
    max_h = info["max_horizon"]
    test_start = info["train_size"] + rolling["val_size"]
    test_end = test_start + rolling["test_size"]

    if rolling["test_size"] < max_h:
        raise ValueError(f"Bloco de teste ({rolling['test_size']}) menor que o horizonte máximo ({max_h}).")
    if test_end > n_rows:
        raise ValueError(f"Blocos treino+val+teste ({test_end}) excedem o tamanho da série ({n_rows}).")

    return list(range(test_start, test_end - max_h + 1, rolling["step_size"]))


PROTOCOL_OPTIONS = ("single_split", "rolling")


def get_protocol(config):
    """Protocolo de avaliação configurado em evaluation.protocol (padrão: single_split)."""
    protocol = config.get("evaluation", {}).get("protocol", "single_split")
    if protocol not in PROTOCOL_OPTIONS:
        raise ValueError(f"evaluation.protocol inválido: '{protocol}'. Use um de {PROTOCOL_OPTIONS}.")
    return protocol


def rolling_input(df, info):
    """
    Prepara a série para cross_validation do Nixtla no protocolo rolling.
    As janelas do Nixtla são contadas a partir do FIM do DataFrame, então a série é cortada
    no fim da última janela; assim as origens coincidem com get_rolling_origins.
    Retorna (df_cortado, origens, kwargs para cross_validation).
    """
    df = df.sort_values("ds").reset_index(drop=True)
    origins = get_rolling_origins(info, len(df))
    df_cut = df.iloc[: origins[-1] + info["max_horizon"]]
    cv_kwargs = {"h": info["max_horizon"], "n_windows": len(origins), "step_size": info["rolling"]["step_size"]}
    return df_cut, origins, cv_kwargs


def check_rolling_cutoffs(df_cv, df, origins, h=None):
    """
    Garante que os cutoffs gerados pelo Nixtla são exatamente os esperados pelo protocolo.
    Com `h`, confere também que cada janela tem as h datas seguintes ao seu cutoff.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    expected = list(df["ds"].iloc[[o - 1 for o in origins]])
    got = sorted(pd.to_datetime(df_cv["cutoff"]).unique())
    if got != expected:
        raise ValueError(
            f"Cutoffs gerados ({len(got)}: {got[:1]}...{got[-1:]}) diferentes dos esperados "
            f"({len(expected)}: {expected[:1]}...{expected[-1:]})."
        )

    if h is not None:
        cv_dates = pd.to_datetime(df_cv["ds"])
        cv_cutoffs = pd.to_datetime(df_cv["cutoff"])
        for start in origins:
            window = sorted(cv_dates[cv_cutoffs == df["ds"].iloc[start - 1]])
            if window != list(df["ds"].iloc[start : start + h]):
                raise ValueError(
                    f"Janela com cutoff {df['ds'].iloc[start - 1]}: datas previstas não correspondem "
                    f"aos {h} passos seguintes da série ({len(window)} datas geradas)."
                )


def clean_series(df, info):
    """
    Limpeza da série na camada bronze: ordena por data, remove datas repetidas, troca valores sentinela
    (info["missing_values"], ex.: -9999) por ausente, completa a grade regular de info["freq"] e
    interpola no tempo os pontos ausentes. Retorna (DataFrame limpo, dict com as contagens).
    """
    date_col, target = info["date_column"], info["target_column"]
    out = df[[date_col, target]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col)

    n_duplicadas = int(out[date_col].duplicated().sum())
    out = out.drop_duplicates(subset=date_col, keep="first")

    sentinelas = info.get("missing_values") or []
    mask = out[target].isin(sentinelas)
    n_sentinelas = int(mask.sum())
    out.loc[mask, target] = np.nan

    grade = pd.date_range(out[date_col].iloc[0], out[date_col].iloc[-1], freq=info["freq"])
    serie = out.set_index(date_col)[target].reindex(grade)
    n_ausentes = int(serie.isna().sum())
    serie = serie.interpolate(method="time", limit_direction="both")

    limpo = serie.rename_axis(date_col).reset_index()
    resumo = {
        "duplicadas": n_duplicadas,
        "sentinelas": n_sentinelas,
        "ausentes_apos_grade": n_ausentes,
        "linhas_antes": len(df),
        "linhas_depois": len(limpo),
    }
    return limpo, resumo


def check_regular_series(df, freq, name):
    """
    Falha com mensagem clara se a série (colunas ds, y) tiver data repetida, buraco, passo irregular ou
    valor ausente. As origens do rolling são calculadas por POSIÇÃO; com datas irregulares, ARIMA, redes
    e híbrido passam a apontar para datas diferentes.
    """
    ds = pd.DatetimeIndex(df["ds"])
    if ds.duplicated().any():
        raise ValueError(f"{name}: {int(ds.duplicated().sum())} data(s) repetida(s) na série (ex.: {ds[ds.duplicated()][0]}).")
    esperado = pd.date_range(ds[0], periods=len(ds), freq=freq)
    if not ds.equals(esperado):
        pos = int(np.argmax(np.asarray(ds != esperado)))
        raise ValueError(
            f"{name}: série fora da grade regular '{freq}' (primeira diferença na posição {pos}: "
            f"{ds[pos]} em vez de {esperado[pos]}). Refaça a ingestão (apague o arquivo bronze)."
        )
    if df["y"].isna().any():
        raise ValueError(f"{name}: {int(df['y'].isna().sum())} valor(es) ausente(s) em y.")


def series_problems(df, info):
    """
    Lista (em texto) o que está errado na série bruta/bronze: datas repetidas, fora da ordem ou da grade
    regular de info["freq"], valores ausentes e valores sentinela. Lista vazia = série pronta para uso.
    """
    problemas = []
    ds = pd.to_datetime(df[info["date_column"]])
    y = df[info["target_column"]]
    if ds.duplicated().any():
        problemas.append(f"{int(ds.duplicated().sum())} data(s) repetida(s)")
    if not ds.is_monotonic_increasing:
        problemas.append("datas fora de ordem")
    else:
        esperado = pd.date_range(ds.iloc[0], periods=len(ds), freq=info["freq"])
        if not pd.DatetimeIndex(ds).equals(esperado):
            problemas.append(f"fora da grade regular '{info['freq']}' (buracos ou passo irregular)")
    if y.isna().any():
        problemas.append(f"{int(y.isna().sum())} valor(es) ausente(s)")
    sentinelas = info.get("missing_values") or []
    if sentinelas and y.isin(sentinelas).any():
        problemas.append(f"{int(y.isin(sentinelas).sum())} valor(es) sentinela {sentinelas}")
    return problemas


def prepare_data(df, info, name):
    """
    Padroniza os dados brutos para o formato exigido pelo Nixtla (StatsForecast/NeuralForecast)
    e confere que a série é regular (sem datas repetidas, buracos ou valores ausentes).
    """
    df_n = df[[info["date_column"], info["target_column"]]].copy()
    df_n.columns = ["ds", "y"]
    df_n["unique_id"] = name
    df_n["ds"] = pd.to_datetime(df_n["ds"])
    check_regular_series(df_n, info["freq"], name)
    return df_n


if __name__ == "__main__":
    # Usado pelo orquestrador: imprime, na última linha, os datasets selecionados no protocolo rolling
    print(" ".join(get_datasets("rolling")))
