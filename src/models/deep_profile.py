"""
Perfil de hiperparâmetros das redes neurais (MLP, LSTM, N-HiTS, Informer e o N-HiTS dos híbridos).

Tudo que é configurável fica em `deep_learning` (config/main_config.yaml); este módulo aplica os padrões,
valida a configuração e monta os modelos do NeuralForecast, escolhendo GPU ou CPU (src/utils/compute.py).

Estratégia de horizonte (`deep_learning.strategy`):
- per_horizon (padrão): um modelo MIMO por comprimento de previsão H (24, 48, 168, 336, 720), como em
  Informer (Zhou et al., 2021), PatchTST (Nie et al., 2023) e DLinear (Zeng et al., 2023). Cada modelo
  prevê H passos de uma vez; NÃO é a estratégia Direct de Ben Taieb et al. (2012), que treina um modelo por passo.
- single_truncated: um único modelo com H = max_horizon, cujos primeiros h passos são usados nos horizontes menores.
"""

import sys
from copy import deepcopy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.utils.compute import get_trainer_kwargs

STRATEGY_OPTIONS = ("per_horizon", "single_truncated")
LOSS_OPTIONS = ("mse", "mae")
MODEL_NAMES = ("MLP", "LSTM", "NHITS", "Informer")

# Padrões usados quando a seção `deep_learning` não existe (ou está incompleta) no main_config.yaml.
# Origem de cada valor:
#   input_size 336 ........ lookback do PatchTST/DLinear no ETTh1 (Informer escolhe entre 24 e 720)
#   loss mse .............. Informer, PatchTST e DLinear treinam com MSE (e a métrica principal é o MSE)
#   max_steps 1000, lr 1e-3  padrão do NeuralForecast para MLP e N-HiTS (também o do tutorial N-HiTS)
#   Informer .............. Zhou et al. (2021): d_model 512, 8 cabeças, 3 camadas no encoder e 2 no decoder,
#                           fator de amostragem 5, dropout 0.1, Adam com lr 1e-4 e lotes de 32 janelas
#   early stopping ........ escolha deste projeto (paciência de 5 validações = 500 passos sem melhora)
DEFAULT_DEEP_CONFIG = {
    "strategy": "per_horizon",
    "input_size": 336,
    "common": {
        "loss": "mse",
        "scaler_type": "standard",
        "max_steps": 1000,
        "val_check_steps": 100,
        "early_stop_patience_steps": 5,
    },
    "models": {
        "MLP": {},
        "LSTM": {"encoder_hidden_size": 64},
        "NHITS": {"learning_rate": 0.001},
        "Informer": {
            "learning_rate": 0.0001,
            "hidden_size": 512,
            "n_head": 8,
            "encoder_layers": 3,
            "decoder_layers": 2,
            "factor": 5,
            "dropout": 0.1,
            "windows_batch_size": 32,
            "max_steps": 2000,
        },
    },
}


def _merge(base, override):
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_deep_config(config=None):
    """Seção `deep_learning` do main_config.yaml sobre os padrões, já validada."""
    cfg = _merge(DEFAULT_DEEP_CONFIG, (config or {}).get("deep_learning"))

    if cfg["strategy"] not in STRATEGY_OPTIONS:
        raise ValueError(f"deep_learning.strategy inválida: '{cfg['strategy']}'. Use um de {STRATEGY_OPTIONS}.")
    if str(cfg["common"]["loss"]).lower() not in LOSS_OPTIONS:
        raise ValueError(f"deep_learning.common.loss inválida: '{cfg['common']['loss']}'. Use um de {LOSS_OPTIONS}.")
    if int(cfg["input_size"]) <= 0:
        raise ValueError("deep_learning.input_size deve ser positivo.")
    unknown = set(cfg["models"]) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"deep_learning.models tem modelos desconhecidos: {sorted(unknown)}. Use {MODEL_NAMES}.")
    return cfg


def model_horizons(info, deep_cfg):
    """
    Comprimentos H dos modelos a treinar. per_horizon: todos os horizontes avaliados do dataset;
    single_truncated: só o max_horizon.
    """
    if deep_cfg["strategy"] == "single_truncated":
        return [info["max_horizon"]]
    horizons = sorted(set(info["forecast_horizon"]) | {info["max_horizon"]})
    return [h for h in horizons if h <= info["max_horizon"]]


def pick_model_horizon(available, h):
    """Menor H treinado que cobre o horizonte avaliado `h` (H = h em per_horizon; max_horizon em single_truncated)."""
    candidates = sorted(x for x in available if x >= h)
    if not candidates:
        raise ValueError(f"Nenhum modelo com H >= {h} (H disponíveis: {sorted(available)}).")
    return candidates[0]


def _loss(name):
    from neuralforecast.losses.pytorch import MAE, MSE

    return {"mse": MSE, "mae": MAE}[str(name).lower()]()


def build_models(horizon, config=None, input_size=None, names=None, extra_kwargs=None):
    """
    Modelos do NeuralForecast para um comprimento de previsão `horizon` (H).
    - `config`: dict do main_config.yaml (seções `deep_learning`, `compute` e `random_seed`).
    - `input_size`: sobrescreve `deep_learning.input_size` (usado nos testes).
    - `names`: subconjunto de MODEL_NAMES (padrão: todos).
    - `extra_kwargs`: kwargs adicionais do Trainer/modelo (ex.: silenciar logs nos testes); têm prioridade.
    """
    from neuralforecast.models import LSTM, MLP, NHITS, Informer

    classes = {"MLP": MLP, "LSTM": LSTM, "NHITS": NHITS, "Informer": Informer}
    cfg = load_deep_config(config)
    input_size = int(input_size or cfg["input_size"])
    names = list(names or MODEL_NAMES)

    runtime = {
        "num_sanity_val_steps": 0,
        "enable_progress_bar": True,
        "num_workers_loader": 0,
        "random_seed": (config or {}).get("random_seed", 42),
        **get_trainer_kwargs(config),
    }
    runtime.update(extra_kwargs or {})

    models = {}
    for name in names:
        params = _merge(cfg["common"], cfg["models"].get(name, {}))
        params["loss"] = _loss(params["loss"])
        models[name] = classes[name](h=horizon, input_size=input_size, **params, **runtime)
    return models


def build_nhits(horizon, config=None, input_size=None, extra_kwargs=None):
    """N-HiTS com o mesmo perfil da baseline pura (também usado nos resíduos dos híbridos ARIMA e Prophet)."""
    return build_models(horizon, config, input_size=input_size, names=["NHITS"], extra_kwargs=extra_kwargs)["NHITS"]
