"""
Otimização de hiperparâmetros (Optuna) das redes neurais: MLP, LSTM, N-HiTS, Informer e o N-HiTS dos resíduos
dos híbridos (chave NHITS_RES).

Protocolo (sem tocar no teste):
- Uma busca por (dataset, modelo), em um horizonte de referência (`hpo.tuning_horizon`; padrão: o do meio da lista).
- Cada tentativa treina SÓ no bloco de treino (early stopping em uma parte final desse bloco) e é avaliada em
  janelas rolantes do bloco de VALIDAÇÃO. Os blocos de teste nunca são lidos.
- Critério: MSE de validação (normalizado pela variância do treino). Para NHITS_RES é o MSE de validação do híbrido
  ARIMA + alpha * N-HiTS (o mesmo critério que escolhe a janela dos resíduos em nhits_residual.py).
- Mesmo orçamento (n_trials) para todos os modelos, salvo `hpo.n_trials_by_model` (declarado no YAML).
- A primeira tentativa é a configuração atual do projeto (a de referência), então o resultado otimizado nunca é pior
  que ela na validação e a comparação "padrão x otimizado" fica registrada.
- Sampler TPE com a semente global; estudo persistido em SQLite (retoma de onde parou).

Evidência gravada em results/hpo/:
  {dataset}_{modelo}.db             estudo Optuna (todas as tentativas, retomável)
  {dataset}_{modelo}_trials.csv     uma linha por tentativa: parâmetros, MSE de validação, duração, estado
  {dataset}_{modelo}_best.json      configuração congelada (usada pelos treinos finais) + metadados

Uso: PF_DATASETS=ETTh1 python src/models/hpo.py --models MLP LSTM NHITS Informer NHITS_RES [--trials 20]
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.models.deep_profile import (
    HPO_MODEL_KEYS,
    build_models,
    hpo_best_path,
    hpo_dir,
    load_deep_config,
    load_hpo_config,
)
from src.utils.general import get_datasets, get_protocol, prepare_data

warnings.filterwarnings("ignore")

# Valores do projeto que não estão no YAML (padrões do NeuralForecast 1.7.7), para a tentativa de referência.
NF_DEFAULTS = {
    "MLP": {"hidden_size": 512, "num_layers": 2, "learning_rate": 1e-3},
    "LSTM": {"encoder_hidden_size": 64, "encoder_n_layers": 2, "encoder_dropout": 0.0, "learning_rate": 1e-3},
    "NHITS": {"mlp_width": 512, "dropout_prob_theta": 0.0, "learning_rate": 1e-3},
    "Informer": {"hidden_size": 512, "dropout": 0.1, "learning_rate": 1e-4},
}
BASE_MODEL = {"MLP": "MLP", "LSTM": "LSTM", "NHITS": "NHITS", "Informer": "Informer", "NHITS_RES": "NHITS"}

# Espaço de busca declarado (documenta o que foi otimizado). Valores categóricos incluem o padrão do projeto.
SEARCH_SPACE = {
    "MLP": {
        "input_size": [24, 96, 336],
        "learning_rate": "loguniform[1e-4, 1e-2]",
        "hidden_size": [128, 256, 512],
        "num_layers": "int[2, 4]",
    },
    "LSTM": {
        "input_size": [24, 96, 336],
        "learning_rate": "loguniform[1e-4, 1e-2]",
        "encoder_hidden_size": [32, 64, 128, 256],
        "encoder_n_layers": "int[1, 2]",
        "encoder_dropout": [0.0, 0.1, 0.3],
    },
    "NHITS": {
        "input_size": [24, 96, 336],
        "learning_rate": "loguniform[1e-4, 3e-3]",
        "mlp_width": [256, 512],
        "dropout_prob_theta": [0.0, 0.1, 0.2],
    },
    "Informer": {
        "input_size": [96, 336],
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "hidden_size": [128, 256, 512],
        "dropout": [0.05, 0.1, 0.2],
    },
}
SEARCH_SPACE["NHITS_RES"] = SEARCH_SPACE["NHITS"]


def suggest_params(key, trial):
    """Sorteia os parâmetros da tentativa e devolve os kwargs prontos para o modelo do NeuralForecast."""
    if key == "MLP":
        return {
            "input_size": trial.suggest_categorical("input_size", SEARCH_SPACE[key]["input_size"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", SEARCH_SPACE[key]["hidden_size"]),
            "num_layers": trial.suggest_int("num_layers", 2, 4),
        }
    if key == "LSTM":
        layers = trial.suggest_int("encoder_n_layers", 1, 2)
        dropout = trial.suggest_categorical("encoder_dropout", SEARCH_SPACE[key]["encoder_dropout"])
        return {
            "input_size": trial.suggest_categorical("input_size", SEARCH_SPACE[key]["input_size"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "encoder_hidden_size": trial.suggest_categorical("encoder_hidden_size", SEARCH_SPACE[key]["encoder_hidden_size"]),
            "encoder_n_layers": layers,
            "encoder_dropout": dropout if layers > 1 else 0.0,  # dropout entre camadas só existe com 2+ camadas
        }
    if key in ("NHITS", "NHITS_RES"):
        width = trial.suggest_categorical("mlp_width", SEARCH_SPACE[key]["mlp_width"])
        return {
            "input_size": trial.suggest_categorical("input_size", SEARCH_SPACE[key]["input_size"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "mlp_units": [[width, width]] * 3,
            "dropout_prob_theta": trial.suggest_categorical("dropout_prob_theta", SEARCH_SPACE[key]["dropout_prob_theta"]),
        }
    if key == "Informer":
        return {
            "input_size": trial.suggest_categorical("input_size", SEARCH_SPACE[key]["input_size"]),
            "learning_rate": trial.suggest_categorical("learning_rate", SEARCH_SPACE[key]["learning_rate"]),
            "hidden_size": trial.suggest_categorical("hidden_size", SEARCH_SPACE[key]["hidden_size"]),
            "dropout": trial.suggest_categorical("dropout", SEARCH_SPACE[key]["dropout"]),
        }
    raise ValueError(f"Modelo desconhecido para HPO: {key}. Use {HPO_MODEL_KEYS}.")


def reference_params(key, config, deep_cfg, hybrid_input_size=None):
    """Tentativa de referência: a configuração atual do projeto, na forma dos parâmetros sorteados (enqueue_trial)."""
    base = BASE_MODEL[key]
    merged = {**NF_DEFAULTS[base], **{k: v for k, v in deep_cfg["models"].get(base, {}).items() if k in NF_DEFAULTS[base]}}
    ref = {"input_size": int(hybrid_input_size or deep_cfg["input_size"])}
    if base == "NHITS":
        ref.update(learning_rate=merged["learning_rate"], mlp_width=merged["mlp_width"], dropout_prob_theta=merged["dropout_prob_theta"])
    else:
        ref.update({k: merged[k] for k in NF_DEFAULTS[base]})
    return ref


def tuning_horizon(info, deep_cfg, hpo_cfg):
    """Horizonte H da busca: o configurado ou o do meio da lista. Precisa caber no bloco de validação."""
    from src.models.deep_profile import model_horizons

    horizons = model_horizons(info, deep_cfg)
    h = int(hpo_cfg["tuning_horizon"]) if hpo_cfg["tuning_horizon"] else horizons[len(horizons) // 2]
    val_size = info["rolling"]["val_size"]
    if h > val_size:
        fitting = [x for x in horizons if x <= val_size]
        if not fitting:
            raise ValueError(f"Nenhum horizonte cabe no bloco de validação ({val_size}).")
        h = max(fitting)
    return h


def inner_val_size(info, h):
    """Tamanho do early stopping dentro do bloco de treino (nunca menor que H)."""
    train_end = info["train_size"]
    return max(h, min(info["rolling"]["val_size"], train_end // 4))


def validation_starts(info, h):
    """Origens das janelas de H passos que cabem inteiras no bloco de validação (mesmo passo do protocolo rolling)."""
    train_end = info["train_size"]
    val_end = train_end + info["rolling"]["val_size"]
    return list(range(train_end, val_end - h + 1, info["rolling"]["step_size"]))


class Objective:
    """Treina no bloco de treino e devolve o MSE no bloco de validação (menor é melhor)."""

    def __init__(self, key, name, info, df, config, h, trial_max_steps=None, df_res=None, df_arima_val=None, alpha_step=0.05):
        self.key, self.name, self.info, self.df, self.config, self.h = key, name, info, df, config, h
        self.trial_max_steps = trial_max_steps
        self.df_res, self.df_arima_val, self.alpha_step = df_res, df_arima_val, alpha_step
        self.starts = validation_starts(info, h)
        if key == "NHITS_RES":
            # As previsões do ARIMA na validação cobrem janelas de max_horizon passos: usa-se as mesmas origens do alpha
            from src.models.arima_baseline import validation_origins

            self.starts = validation_origins(info, len(df))
        if not self.starts:
            raise ValueError(f"{name}: nenhuma janela de H={h} cabe no bloco de validação.")
        train = df["y"].iloc[: info["train_size"]]
        self.var_train = float(train.var()) or 1.0

    def kwargs(self, params):
        kw = dict(params)
        if self.trial_max_steps:
            kw["max_steps"] = int(self.trial_max_steps)
        return kw

    def score(self, params):
        from neuralforecast import NeuralForecast

        base = BASE_MODEL[self.key]
        kw = self.kwargs(params)
        input_size = int(kw["input_size"])
        runtime = {"enable_progress_bar": False, "enable_model_summary": False, "logger": False, "enable_checkpointing": False}
        model = build_models(self.h, self.config, names=[base], extra_kwargs=runtime, overrides={base: kw})[base]

        series = self.df_res if self.key == "NHITS_RES" else self.df
        series = series.sort_values("ds").reset_index(drop=True)
        train_end = self.info["train_size"]
        if train_end < input_size + self.h:
            raise ValueError(f"Treino ({train_end}) curto para input_size={input_size} e H={self.h}.")
        df_fit = series.iloc[:train_end]
        if df_fit["y"].isna().any():
            raise ValueError("Valores ausentes no bloco de treino usado na busca.")

        nf = NeuralForecast(models=[model], freq=self.info["freq"])
        nf.fit(df=df_fit, val_size=inner_val_size(self.info, self.h))

        if self.key == "NHITS_RES":
            from src.models.hybrid_variants import best_val_mse
            from src.models.nhits_residual import nhits_validation_predictions

            preds = nhits_validation_predictions(nf, series, input_size, self.h, self.starts)
            if preds.empty:
                raise ValueError("Sem previsões de validação para a rede dos resíduos.")
            return float(best_val_mse(self.df_arima_val, preds, self.alpha_step)) / self.var_train

        errors = []
        for start in self.starts:
            context = series.iloc[start - input_size : start]
            true = series["y"].iloc[start : start + self.h].to_numpy()
            future = nf.predict(df=context).reset_index().sort_values("ds")
            pred = future[base].to_numpy()[: self.h]
            if len(pred) != self.h or len(true) != self.h or np.isnan(true).any():
                continue
            errors.append(np.mean((pred - true) ** 2))
        if not errors:
            raise ValueError("Sem janelas de validação válidas.")
        return float(np.mean(errors)) / self.var_train

    def __call__(self, trial):
        from src.models.deep_baselines import free_memory

        params = suggest_params(self.key, trial)
        try:
            value = self.score(params)
        finally:
            free_memory()
        if not np.isfinite(value):
            raise ValueError("MSE de validação não finito.")
        return value


def _git_commit():
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def tune(config, name, info, key, n_trials=None, quiet=False):
    """Roda (ou retoma) a busca de (dataset, modelo) e grava trials.csv e best.json. Devolve o dict do best.json."""
    import optuna

    hpo_cfg = load_hpo_config(config)
    deep_cfg = load_deep_config(config)
    seed = int(config.get("random_seed", 42))
    n_trials = int(n_trials or hpo_cfg["n_trials_by_model"].get(key, hpo_cfg["n_trials"]))
    out_dir = hpo_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_data(pd.read_csv(BASE_DIR / info["path"]), info, name).sort_values("ds").reset_index(drop=True)
    h = tuning_horizon(info, deep_cfg, hpo_cfg)

    df_res = df_arima_val = None
    hybrid_input = None
    alpha_step = 0.05
    if key == "NHITS_RES":
        from src.models.hybrid_variants import load_hybrid_config
        from src.models.nhits_residual import residuals_on_full_dates

        hybrid_cfg = load_hybrid_config(config)
        alpha_step = hybrid_cfg["alpha"]["grid_step"]
        hybrid_input = hybrid_cfg["residual_input_size"] or deep_cfg["input_size"]
        resid_path = BASE_DIR / config["results_paths"]["residuals"] / f"{name}_arima_residuals_rolling.csv"
        arima_val = BASE_DIR / config["results_paths"]["arima"] / f"{name}_arima_val_rolling.csv"
        if not resid_path.exists() or not arima_val.exists():
            raise FileNotFoundError(f"HPO de NHITS_RES exige o ARIMA rolling de {name}: faltam {resid_path.name} e/ou {arima_val.name}.")
        df_res = residuals_on_full_dates(pd.read_csv(resid_path, parse_dates=["ds"]), df)
        df_arima_val = pd.read_csv(arima_val, parse_dates=["cutoff", "ds"])

    objective = Objective(key, name, info, df, config, h, hpo_cfg["trial_max_steps"], df_res, df_arima_val, alpha_step)

    study_name = f"{name}_{key}"
    storage = f"sqlite:///{(out_dir / (study_name + '.db')).as_posix()}"
    optuna.logging.set_verbosity(optuna.logging.WARNING if quiet else optuna.logging.INFO)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    if not study.trials:
        study.enqueue_trial(reference_params(key, config, deep_cfg, hybrid_input))

    done = len([t for t in study.trials if t.state.name in ("COMPLETE", "FAIL")])
    remaining = max(0, n_trials - done)
    timeout = hpo_cfg["timeout_minutes"] * 60 if hpo_cfg["timeout_minutes"] else None
    print(f"\n🔎 HPO {name}/{key}: H={h}, {len(objective.starts)} janelas de validação, {remaining} tentativas a rodar (alvo {n_trials}).")
    start = time.time()
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=timeout, catch=(Exception,), gc_after_trial=True)
    elapsed = time.time() - start

    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not complete:
        raise RuntimeError(f"HPO {name}/{key}: nenhuma tentativa concluída (veja {study_name}_trials.csv).")

    trials = study.trials_dataframe()
    trials.to_csv(out_dir / f"{study_name}_trials.csv", index=False)

    best = study.best_trial
    ref_value = study.trials[0].value if study.trials[0].state.name == "COMPLETE" else None
    best_params = suggest_params(key, optuna.trial.FixedTrial(best.params))  # sem o max_steps reduzido da busca
    result = {
        "dataset": name,
        "model": key,
        "tuning_horizon": h,
        "validation_windows": len(objective.starts),
        "criterion": "MSE de validação / var(treino)" + (" do híbrido ARIMA + alpha*N-HiTS" if key == "NHITS_RES" else ""),
        "best_value": best.value,
        "reference_value": ref_value,
        "improvement_vs_reference_pct": (100.0 * (ref_value - best.value) / ref_value) if ref_value else None,
        "n_trials_completed": len(complete),
        "n_trials_failed": len([t for t in study.trials if t.state.name == "FAIL"]),
        "params": best_params,
        "search_space": SEARCH_SPACE[key],
        "seed": seed,
        "trial_max_steps": hpo_cfg["trial_max_steps"],
        "optuna_version": optuna.__version__,
        "commit": _git_commit(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "last_run_seconds": round(elapsed, 1),
    }
    with open(hpo_best_path(config, name, key), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(
        f"✅ HPO {name}/{key}: melhor MSE de validação normalizado = {best.value:.5f} "
        f"(referência {ref_value if ref_value is None else round(ref_value, 5)}) -> {hpo_best_path(config, name, key).name}"
    )
    print(f"   Parâmetros: {best_params}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Otimização de hiperparâmetros (Optuna) das redes neurais.")
    parser.add_argument("--models", nargs="+", default=None, help=f"Subconjunto de {HPO_MODEL_KEYS}.")
    parser.add_argument("--trials", type=int, default=None, help="Sobrescreve hpo.n_trials / n_trials_by_model.")
    parser.add_argument("--enabled", action="store_true", help="Imprime 1 se hpo.enabled estiver ligado no YAML, senão 0.")
    args = parser.parse_args()

    with open(BASE_DIR / "config" / "main_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.enabled:
        print(1 if load_hpo_config(config)["enabled"] else 0)
        return

    models = list(args.models or HPO_MODEL_KEYS)
    unknown = set(models) - set(HPO_MODEL_KEYS)
    if unknown:
        raise SystemExit(f"Modelos desconhecidos: {sorted(unknown)}. Use {HPO_MODEL_KEYS}.")
    protocol = get_protocol(config)
    if protocol != "rolling":
        raise SystemExit("A otimização de hiperparâmetros exige evaluation.protocol: rolling.")

    failed = []
    for name, info in get_datasets(protocol).items():
        for key in models:
            try:
                tune(config, name, info, key, n_trials=args.trials)
            except Exception as exc:  # um modelo que falha não impede os demais; o código de saída acusa
                failed.append(f"{name}/{key}")
                print(f"❌ HPO {name}/{key} falhou: {exc}")
    if failed:
        raise SystemExit(f"HPO incompleto: {', '.join(failed)}")


if __name__ == "__main__":
    main()
