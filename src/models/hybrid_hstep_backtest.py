"""
Híbrido ARIMA–N-HiTS com correção do erro de h passos: o N-HiTS enxerga o trecho completo do bloco (série original + previsão do ARIMA)
e é treinado no erro REAL de h passos do ARIMA, obtido num backtest rolling fora da amostra.

Diferenças em relação ao híbrido aditivo (hybrid_combiner.py):

| ponto                | híbrido aditivo atual                  | híbrido erro h (este módulo)                          |
|----------------------|----------------------------------------|-------------------------------------------------------|
| alvo da rede         | resíduo in-sample de 1 passo           | erro de h passos do ARIMA numa origem (backtest)       |
| entrada da rede      | só a história dos resíduos             | série original (passado) + previsão do ARIMA (futuro) |
| treino x aplicação   | alvo (1 passo) != o que corrige (h)    | alvo = exatamente o que será corrigido no teste        |

Construção dos dados (cada origem do backtest vira uma série curta de tamanho input_size + H no NeuralForecast,
e portanto exatamente uma janela de treino):

- exógena futura `path`: nos `input_size` passos de contexto vale a série original y (z-score, o que o ARIMA
  também vê); nos H passos seguintes vale a previsão de h passos do ARIMA feita naquela origem;
- alvo `y`: 0 no contexto (a rede não usa o próprio alvo passado; ela "lê" `path`) e, nos H passos, o erro
  y_real - ARIMA daquela origem. O NHITS roda sem scaler (a história do alvo é constante).

Backtest do ARIMA: o AutoARIMA é ajustado nos primeiros `fit_frac` do bloco de treino e, com parâmetros fixos,
atualiza o estado a cada `stride` passos (cross_validation refit=False). Assim cada previsão do backtest é fora
da amostra, também em relação aos parâmetros.

Sem vazamento / sem otimismo na escolha do alpha:
- rede A: treinada só com origens cujo alvo termina antes da validação; prevê as origens de VALIDAÇÃO
  (as mesmas usadas pelo alpha dos outros híbridos) -> alpha por H com previsões genuinamente fora da amostra;
- rede B: treinada com origens até o fim da validação (sem early stopping, mesmos max_steps); prevê o TESTE.
O teste nunca entra no treino. As previsões do ARIMA nas origens de validação/teste são as dos arquivos
`*_arima_val_rolling.csv` e `*_arima_rolling.csv`, as mesmas dos demais híbridos.

Saídas: {dataset}_hybrid_hstep_rolling.csv (colunas Hybrid_ARIMA_NHITS_hstep e Hybrid_ARIMA_NHITS_hstep_alpha, com H),
{dataset}_hybrid_hstep_alpha.csv (alpha, MSE de validação, habilidade da correção) e o backtest em
{dataset}_hybrid_hstep_backtest.csv.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.models.arima_baseline import validation_origins
from src.models.hybrid_variants import KEYS, alpha_by_horizon, load_hybrid_config, residual_input_sizes
from src.utils.general import check_rolling_cutoffs, get_datasets, get_protocol, get_rolling_origins, prepare_data

warnings.filterwarnings("ignore")

HSTEP_COL = "Hybrid_ARIMA_NHITS_hstep"
HSTEP_ALPHA_COL = "Hybrid_ARIMA_NHITS_hstep_alpha"
EXOG_NAME = "path"

DEFAULT_HSTEP_CONFIG = {
    "enabled": True,
    "fit_frac": 0.5,  # fração do bloco de treino usada para ajustar os parâmetros do ARIMA do backtest
    "stride": None,  # passo entre origens do backtest; None -> rolling.step_size do dataset
    "max_steps": None,  # None -> usa os max_steps da rede dos resíduos (NHITS_RES)
}


def load_hstep_config(config):
    """Seção `hybrid_hstep` do YAML com padrões; valida os valores."""
    cfg = {**DEFAULT_HSTEP_CONFIG, **((config or {}).get("hybrid_hstep") or {})}
    if not 0.1 <= float(cfg["fit_frac"]) < 1:
        raise ValueError(f"hybrid_hstep.fit_frac deve estar em [0.1, 1), recebido {cfg['fit_frac']}.")
    if cfg["stride"] is not None and int(cfg["stride"]) <= 0:
        raise ValueError("hybrid_hstep.stride deve ser um inteiro positivo.")
    return cfg


# ----------------------------------------------------------------------------
# Backtest do ARIMA (fora da amostra) no treino e na validação
# ----------------------------------------------------------------------------
def backtest_origins(info, fit_frac, stride):
    """Origens (posição do 1º passo previsto) do backtest: do fim do ajuste até o fim da validação - H máximo."""
    fit_end = int(info["train_size"] * float(fit_frac))
    val_end = info["train_size"] + info["rolling"]["val_size"]
    origins = list(range(fit_end, val_end - info["max_horizon"] + 1, int(stride)))
    if not origins:
        raise ValueError("Backtest sem origens: treino+validação menores que o ajuste + horizonte máximo.")
    return origins


def arima_backtest(df, info, fit_frac=0.5, stride=None, n_jobs=-1):
    """
    Previsões de max_horizon passos do ARIMA nas origens do backtest, com parâmetros ajustados só no começo do
    treino. Retorna unique_id, cutoff, ds, y, ARIMA (formato do cross_validation).
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    df = df.sort_values("ds").reset_index(drop=True)
    stride = int(stride or info["rolling"]["step_size"])
    origins = backtest_origins(info, fit_frac, stride)
    max_h = info["max_horizon"]
    df_cut = df.iloc[: origins[-1] + max_h]
    sf = StatsForecast(
        models=[AutoARIMA(season_length=info["seasonal_period"], approximation=True)], freq=info["freq"], n_jobs=n_jobs
    )
    cv = sf.cross_validation(df=df_cut, h=max_h, n_windows=len(origins), step_size=stride, refit=False).reset_index()
    cv = cv.rename(columns={"AutoARIMA": "ARIMA"})
    check_rolling_cutoffs(cv, df, origins, h=max_h)
    return cv[["unique_id", "cutoff", "ds", "y", "ARIMA"]]


# ----------------------------------------------------------------------------
# Montagem dos dados da rede
# ----------------------------------------------------------------------------
def _origin_series(df, forecast_by_cutoff, pos, input_size, h, with_target):
    """Série curta de uma origem `pos`: contexto (y) + H passos (previsão do ARIMA e, no treino, o erro)."""
    ctx = df.iloc[pos - input_size : pos]
    fut_ds = df["ds"].iloc[pos : pos + h]
    fc = forecast_by_cutoff.get(df["ds"].iloc[pos - 1])
    if fc is None or len(fc) < h:
        raise ValueError(f"Sem previsão do ARIMA com {h} passos na origem {pos}.")
    fc = fc.iloc[:h]
    if not (fc["ds"].to_numpy() == fut_ds.to_numpy()).all():
        raise ValueError(f"Datas da previsão do ARIMA divergem das datas reais na origem {pos}.")
    uid = f"o{pos}"
    hist = pd.DataFrame(
        {"unique_id": uid, "ds": ctx["ds"].to_numpy(), "y": 0.0, EXOG_NAME: ctx["y"].to_numpy(dtype=float)}
    )
    fut = pd.DataFrame(
        {
            "unique_id": uid,
            "ds": fut_ds.to_numpy(),
            "y": (fc["y"].to_numpy() - fc["ARIMA"].to_numpy()) if with_target else 0.0,
            EXOG_NAME: fc["ARIMA"].to_numpy(dtype=float),
        }
    )
    return hist, fut


def _by_cutoff(forecasts):
    return {c: g.sort_values("ds").reset_index(drop=True) for c, g in forecasts.groupby("cutoff")}


def training_frame(df, backtest, positions, input_size, h):
    """Uma série (contexto + H) por origem em `positions`, com o erro de h passos do ARIMA como alvo."""
    by_cut = _by_cutoff(backtest)
    parts = []
    for pos in positions:
        hist, fut = _origin_series(df, by_cut, pos, input_size, h, with_target=True)
        parts.extend([hist, fut])
    out = pd.concat(parts, ignore_index=True)
    return out


def inference_frames(df, forecasts, positions, input_size, h):
    """(histórico, futuro) para prever a correção nas origens `positions` com as previsões do ARIMA dadas."""
    by_cut = _by_cutoff(forecasts)
    hists, futs = [], []
    for pos in positions:
        hist, fut = _origin_series(df, by_cut, pos, input_size, h, with_target=False)
        hists.append(hist)
        futs.append(fut[["unique_id", "ds", EXOG_NAME]])
    return pd.concat(hists, ignore_index=True), pd.concat(futs, ignore_index=True)


def positions_for_horizon(backtest_positions, boundary, h):
    """Origens do backtest cujo alvo de h passos termina antes de `boundary` (fim do treino ou da validação)."""
    return [p for p in backtest_positions if p + h <= boundary]


def hstep_nhits(h, config, input_size, dataset, max_steps=None):
    """N-HiTS com a exógena `path`, sem scaler e sem early stopping (não há validação interna)."""
    from src.models.deep_profile import build_nhits

    overrides = {
        "futr_exog_list": [EXOG_NAME],
        "scaler_type": None,
        "early_stop_patience_steps": -1,
        "input_size": input_size,
    }
    if max_steps:
        overrides["max_steps"] = int(max_steps)
    return build_nhits(h, config, input_size, dataset=dataset, overrides=overrides)


def predict_corrections(nf, df, forecasts, positions, input_size, h):
    """Correção prevista pela rede para cada origem: cutoff, ds, NHITS (H passos cada)."""
    hist, futr = inference_frames(df, forecasts, positions, input_size, h)
    pred = nf.predict(df=hist, futr_df=futr).reset_index()
    col = [c for c in pred.columns if c not in ("unique_id", "ds")][0]
    pred = pred.rename(columns={col: "NHITS"})
    pos_by_uid = {f"o{p}": df["ds"].iloc[p - 1] for p in positions}
    pred["cutoff"] = pred["unique_id"].map(pos_by_uid)
    if pred["cutoff"].isna().any() or len(pred) != h * len(positions):
        raise ValueError(f"Previsão da rede com {len(pred)} linhas; esperado {h * len(positions)} (H={h}).")
    return pred[["cutoff", "ds", "NHITS"]].sort_values(KEYS).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Pipeline por dataset
# ----------------------------------------------------------------------------
def hybrid_hstep_rolling(df, info, config, dataset, df_arima, df_arima_val, backtest, horizons, input_size, hstep_cfg,
                        alpha_step=0.05, model_factory=None):
    """
    Retorna (previsões do teste com H, tabela de alpha). `model_factory(h)` devolve o N-HiTS (injetável nos testes).
    """
    from neuralforecast import NeuralForecast

    from src.models.deep_baselines import free_memory

    df = df.sort_values("ds").reset_index(drop=True)
    test_pos = get_rolling_origins(info, len(df))
    val_pos = validation_origins(info, len(df))
    if not val_pos:
        raise ValueError("Validação menor que o horizonte máximo: o híbrido ARIMA+N-HiTS (erro h) precisa dela para o alpha.")
    bt_pos = [int(np.flatnonzero(df["ds"].to_numpy() == c)[0]) + 1 for c in sorted(backtest["cutoff"].unique())]
    train_end = info["train_size"]
    val_end = train_end + info["rolling"]["val_size"]
    factory = model_factory or (lambda h: hstep_nhits(h, config, input_size, dataset, hstep_cfg["max_steps"]))

    val_frames, test_frames = [], []
    for h in horizons:
        pos_a = positions_for_horizon(bt_pos, train_end, h)
        pos_b = positions_for_horizon(bt_pos, val_end, h)
        if len(pos_a) < 2 or len(pos_b) < 2:
            raise ValueError(f"Poucas origens de backtest para H={h} (A={len(pos_a)}, B={len(pos_b)}).")
        print(f"   > H={h}: rede A (treino, {len(pos_a)} origens) -> validação; rede B (treino+val, {len(pos_b)}) -> teste")

        nf_a = NeuralForecast(models=[factory(h)], freq=info["freq"])
        nf_a.fit(df=training_frame(df, backtest, pos_a, input_size, h), val_size=0)
        val = predict_corrections(nf_a, df, df_arima_val, val_pos, input_size, h)
        val.insert(0, "H", h)
        val_frames.append(val)
        del nf_a
        free_memory()

        nf_b = NeuralForecast(models=[factory(h)], freq=info["freq"])
        nf_b.fit(df=training_frame(df, backtest, pos_b, input_size, h), val_size=0)
        test = predict_corrections(nf_b, df, df_arima, test_pos, input_size, h)
        test.insert(0, "H", h)
        test_frames.append(test)
        del nf_b
        free_memory()

    df_val = pd.concat(val_frames, ignore_index=True)
    df_test = pd.concat(test_frames, ignore_index=True)

    table = alpha_by_horizon(df_arima_val, df_val, step=alpha_step)
    table["skill_residual"] = 1.0 - table["mse_val_aditivo"] / table["mse_val_arima"]
    alphas = dict(zip(table["H"], table["alpha"]))

    out = pd.merge(df_test, df_arima[["unique_id"] + KEYS + ["ARIMA"]], on=KEYS, how="left")
    if out["ARIMA"].isna().any():
        raise ValueError("O ARIMA não cobre todas as datas do híbrido ARIMA+N-HiTS (erro h).")
    out["NHITS_hstep"] = out["NHITS"]
    out[HSTEP_COL] = out["ARIMA"] + out["NHITS_hstep"]
    out[HSTEP_ALPHA_COL] = out["ARIMA"] + out["H"].map(alphas).astype(float) * out["NHITS_hstep"]
    out = out[["unique_id", "H", "cutoff", "ds", "ARIMA", "NHITS_hstep", HSTEP_COL, HSTEP_ALPHA_COL]]
    out = out.sort_values(["H", "cutoff", "ds"]).reset_index(drop=True)
    for h, df_h in out.groupby("H"):
        check_rolling_cutoffs(df_h, df, test_pos, h=h)
    return out, table


class HybridHstepBacktestTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        paths = self.config["results_paths"]
        self.arima_dir = BASE_DIR / paths["arima"]
        self.out_dir = BASE_DIR / paths.get("hybrid_hstep", "results/forecasts/hybrid_hstep/")
        self.metrics_dir = BASE_DIR / paths["metrics"]
        self.protocol = get_protocol(self.config)
        self.hstep_cfg = load_hstep_config(self.config)
        self.hybrid_cfg = load_hybrid_config(self.config)

    def run(self):
        from src.models.deep_profile import load_deep_config, model_horizons, tuned_input_size
        from src.utils.compute import describe_compute

        print("\n🧬 Iniciando Treinamento: Híbrido ARIMA–N-HiTS com correção do erro de h passos")
        print(describe_compute(self.config))
        if self.protocol != "rolling":
            print("⚠️ O híbrido ARIMA+N-HiTS (erro h) só existe no protocolo rolling (evaluation.protocol). Nada a fazer.")
            return
        if not self.hstep_cfg["enabled"]:
            print("ℹ️ hybrid_hstep.enabled=false: etapa pulada.")
            return

        deep_cfg = load_deep_config(self.config)
        for name, info in get_datasets(self.protocol).items():
            print(f"\n=======================================\n🧬 Híbrido ARIMA–N-HiTS com correção do erro de h passos para: {name}\n=======================================")
            raw_file = BASE_DIR / info["path"]
            arima_file = self.arima_dir / f"{name}_arima_rolling.csv"
            val_file = self.arima_dir / f"{name}_arima_val_rolling.csv"
            if not (raw_file.exists() and arima_file.exists() and val_file.exists()):
                print(f"⚠️ Faltam dados ou ARIMA (treino/validação) de {name}. Rode o ARIMA antes. Pulando...")
                continue

            df = prepare_data(pd.read_csv(raw_file), info, name)
            df_arima = pd.read_csv(arima_file, parse_dates=["cutoff", "ds"])
            df_arima_val = pd.read_csv(val_file, parse_dates=["cutoff", "ds"])

            input_size = tuned_input_size(self.config, name, "NHITS_RES", residual_input_sizes(self.hybrid_cfg, deep_cfg)[0])
            horizons = model_horizons(info, deep_cfg)
            print(f"   > input_size={input_size} | H={horizons} | fit_frac={self.hstep_cfg['fit_frac']}")

            backtest = arima_backtest(df, info, self.hstep_cfg["fit_frac"], self.hstep_cfg["stride"])
            self.out_dir.mkdir(parents=True, exist_ok=True)
            backtest.to_csv(self.out_dir / f"{name}_hybrid_hstep_backtest.csv", index=False)
            print(f"   > Backtest do ARIMA: {backtest['cutoff'].nunique()} origens fora da amostra")

            out, table = hybrid_hstep_rolling(
                df, info, self.config, name, df_arima, df_arima_val, backtest, horizons, input_size, self.hstep_cfg,
                alpha_step=self.hybrid_cfg["alpha"]["grid_step"],
            )
            out_file = self.out_dir / f"{name}_hybrid_hstep_rolling.csv"
            out.to_csv(out_file, index=False)
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            table_file = self.metrics_dir / f"{name}_hybrid_hstep_alpha.csv"
            table.round(6).to_csv(table_file, index=False)
            print(f"✅ Híbrido ARIMA+N-HiTS (erro h) salvo em: {out_file.parent.name}/{out_file.name}")
            print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    HybridHstepBacktestTrainer().run()
