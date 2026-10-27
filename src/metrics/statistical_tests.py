"""
Testes estatísticos para comparar modelos de previsão (protocolo rolling).

Base metodológica:
- Ben Taieb, Bontempi, Atiya & Sorjamaa (2012), Expert Systems with Applications 39: Friedman +
  Iman-Davenport e pós-teste para comparar várias estratégias (seção 4.2.2).
- Demšar (2006) e García & Herrera (2009): comparação de vários modelos em vários conjuntos de dados.
- Diebold & Mariano (1995) e Harvey, Leybourne & Newbold (1997): comparação par a par da acurácia
  preditiva, com correção para amostras pequenas.

Decisões importantes:
1. As janelas rolantes do ETT se sobrepõem quando o horizonte é maior que o passo entre origens
   (ex.: 720 h com passo de 168 h). Por isso o Diebold-Mariano usa variância de longo prazo com
   `max_lag = ceil(h / passo) - 1` defasagens, e não assume janelas independentes.
2. O Friedman pressupõe blocos independentes. Janelas de uma mesma série NÃO são independentes,
   então com um único dataset o resultado é marcado como exploratório; com vários datasets o
   bloco é o dataset (Demšar recomenda N > 10 blocos).
3. O pós-teste usa Holm (mais simples que o Shaffer usado no artigo; ambos controlam o erro
   familiar). A diferença crítica de Nemenyi é reportada como referência.

Convenção de sinal: `diferenca_media = perda(A) - perda(B)`. Positiva => A errou mais que B.
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

ALPHA = 0.05
LOSS_COLUMNS = ("MSE", "MAE")
MIN_BLOCKS_RULE_OF_THUMB = 10  # Demšar (2006): N > 10


# ==============================================================================
# FUNÇÕES ESTATÍSTICAS (puras, testáveis com dados sintéticos)
# ==============================================================================
def holm_adjust(p_values):
    """Correção de Holm (step-down) para comparações múltiplas. Retorna p-valores ajustados."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def max_lag_for_overlap(horizon, step):
    """Defasagens de autocorrelação entre janelas: janelas a até ceil(h/passo)-1 passos se sobrepõem."""
    if step <= 0:
        raise ValueError("step deve ser positivo.")
    return max(0, math.ceil(horizon / step) - 1)


def diebold_mariano(loss_a, loss_b, max_lag=0):
    """
    Teste de Diebold-Mariano (H0: mesma acurácia esperada) sobre o diferencial de perdas por janela,
    com correção de Harvey-Leybourne-Newbold e p-valor da t de Student (n - 1 g.l.).
    `max_lag` = nº de autocovariâncias incluídas na variância de longo prazo (janelas sobrepostas).
    Retorna dict com n, diferenca_media, estatistica e p_valor (NaN se não for calculável).
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss_a e loss_b precisam ter o mesmo tamanho.")
    d = a - b
    n = len(d)
    out = {"n": n, "diferenca_media": float(np.mean(d)) if n else np.nan, "estatistica": np.nan, "p_valor": np.nan}
    if n < 3 or np.allclose(d, d[0]):
        return out

    dbar = d.mean()
    centered = d - dbar
    long_run = float(np.sum(centered**2) / n)
    for k in range(1, min(max_lag, n - 1) + 1):
        long_run += 2.0 * float(np.sum(centered[k:] * centered[:-k]) / n)
    var_dbar = long_run / n
    if var_dbar <= 0:
        return out

    dm = dbar / math.sqrt(var_dbar)
    h_eff = max_lag + 1  # horizonte efetivo da correção HLN
    factor_arg = (n + 1 - 2 * h_eff + h_eff * (h_eff - 1) / n) / n
    if factor_arg <= 0:
        return out
    stat = dm * math.sqrt(factor_arg)
    out["estatistica"] = float(stat)
    out["p_valor"] = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return out


def friedman_iman_davenport(losses):
    """
    Friedman + correção de Iman-Davenport. `losses`: matriz (N blocos x k modelos), menor = melhor.
    Retorna dict com N, k, ranks médios (array), Q, p do qui-quadrado, F de Iman-Davenport e seu p.
    """
    x = np.asarray(losses, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] < 2:
        raise ValueError("Friedman exige ao menos 2 blocos e 2 modelos.")
    if np.isnan(x).any():
        raise ValueError("A matriz de perdas não pode conter NaN.")
    n, k = x.shape
    ranks = stats.rankdata(x, axis=1)  # empates recebem a média dos postos
    avg = ranks.mean(axis=0)
    q = 12.0 * n / (k * (k + 1)) * (np.sum(avg**2) - k * (k + 1) ** 2 / 4.0)
    p_chi2 = float(stats.chi2.sf(q, k - 1))
    denom = n * (k - 1) - q
    if denom <= 0:  # concordância perfeita entre blocos
        f_id, p_id = np.inf, 0.0
    else:
        f_id = (n - 1) * q / denom
        p_id = float(stats.f.sf(f_id, k - 1, (k - 1) * (n - 1)))
    return {"N": n, "k": k, "ranks_medios": avg, "Q": float(q), "p_chi2": p_chi2, "F_ID": float(f_id), "p_ID": p_id}


def nemenyi_critical_difference(k, n, alpha=ALPHA):
    """Diferença crítica de Nemenyi entre ranks médios (Demšar, 2006)."""
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    return float(q_alpha * math.sqrt(k * (k + 1) / (6.0 * n)))


def posthoc_pairwise(models, avg_ranks, n, alpha=ALPHA):
    """Todos os pares: z = (Ri - Rj)/sqrt(k(k+1)/(6N)), p bilateral e ajuste de Holm."""
    k = len(models)
    se = math.sqrt(k * (k + 1) / (6.0 * n))
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        z = (avg_ranks[i] - avg_ranks[j]) / se
        rows.append({"Modelo_A": models[i], "Modelo_B": models[j], "z": z, "p_valor": float(2 * stats.norm.sf(abs(z)))})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm_adjust(df["p_valor"])
    df["significativo"] = df["p_holm"] < alpha
    return df


# ==============================================================================
# EXECUÇÃO SOBRE OS RESULTADOS DO PROTOCOLO ROLLING
# ==============================================================================
def dm_table(df_windows, horizons, step, models=None, alpha=ALPHA):
    """
    Diebold-Mariano par a par por horizonte e métrica, usando a perda de cada janela.
    `df_windows`: colunas cutoff, Horizonte, Modelo, MSE, MAE. Holm aplicado dentro de cada
    (horizonte, métrica). Só entram janelas em que os DOIS modelos têm previsão.
    """
    rows = []
    for h in horizons:
        lag = max_lag_for_overlap(h, step)
        df_h = df_windows[df_windows["Horizonte"] == h]
        for metric in LOSS_COLUMNS:
            wide = df_h.pivot(index="cutoff", columns="Modelo", values=metric).sort_index()
            names = [m for m in (models or list(wide.columns)) if m in wide.columns]
            block = []
            for a, b in itertools.combinations(names, 2):
                pair = wide[[a, b]].dropna()
                res = diebold_mariano(pair[a], pair[b], max_lag=lag)
                block.append({"Horizonte": h, "Metrica": metric, "Modelo_A": a, "Modelo_B": b, "lags": lag, **res})
            if not block:
                continue
            df_block = pd.DataFrame(block)
            valid = df_block["p_valor"].notna()
            df_block["p_holm"] = np.nan
            if valid.any():
                df_block.loc[valid, "p_holm"] = holm_adjust(df_block.loc[valid, "p_valor"])
            df_block["significativo"] = df_block["p_holm"] < alpha
            rows.append(df_block)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def friedman_tables(blocks_by_horizon, alpha=ALPHA):
    """
    `blocks_by_horizon`: {horizonte: DataFrame (blocos x modelos) de perdas}. Modelos com valores
    ausentes em algum bloco são descartados. Retorna (resumo, ranks, pós-teste).
    """
    summary, ranks, posthoc = [], [], []
    for h, wide in blocks_by_horizon.items():
        wide = wide.dropna(axis=1)
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            continue
        res = friedman_iman_davenport(wide.to_numpy())
        models = list(wide.columns)
        summary.append(
            {
                "Horizonte": h,
                "N_blocos": res["N"],
                "k_modelos": res["k"],
                "Q": res["Q"],
                "p_chi2": res["p_chi2"],
                "F_Iman_Davenport": res["F_ID"],
                "p_Iman_Davenport": res["p_ID"],
                "diferenca_critica_Nemenyi": nemenyi_critical_difference(res["k"], res["N"], alpha),
                "rejeita_H0": res["p_ID"] < alpha,
            }
        )
        for m, r in zip(models, res["ranks_medios"]):
            ranks.append({"Horizonte": h, "Modelo": m, "rank_medio": float(r)})
        ph = posthoc_pairwise(models, res["ranks_medios"], res["N"], alpha)
        ph.insert(0, "Horizonte", h)
        posthoc.append(ph)
    cat = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()  # noqa: E731
    return pd.DataFrame(summary), pd.DataFrame(ranks), cat(posthoc)


def build_friedman_blocks(df_all, metric="MSE"):
    """
    Monta {horizonte: blocos x modelos}. Vários datasets => bloco = dataset (perda média das janelas).
    Um único dataset => bloco = janela (exploratório: janelas dependentes). Retorna (dict, descrição do bloco).
    """
    datasets = sorted(df_all["Dataset"].unique())
    out = {}
    if len(datasets) >= 2:
        for h, df_h in df_all.groupby("Horizonte"):
            wide = df_h.groupby(["Dataset", "Modelo"])[metric].mean().unstack("Modelo")
            if len(wide) >= 2:
                out[h] = wide
        return out, "dataset"
    for h, df_h in df_all.groupby("Horizonte"):
        out[h] = df_h.pivot(index="cutoff", columns="Modelo", values=metric)
    return out, "janela (exploratório: janelas dependentes)"


def run_dm_for_dataset(dataset, info, df_windows, metrics_dir):
    """Diebold-Mariano de um dataset; salva {dataset}_dm_tests.csv em `metrics_dir`."""
    step = info["rolling"]["step_size"]
    horizons = info.get("forecast_horizon")
    df_dm = dm_table(df_windows, horizons, step)
    out = Path(metrics_dir) / f"{dataset}_dm_tests.csv"
    df_dm.round(6).to_csv(out, index=False)
    n_sig = int(df_dm["significativo"].sum()) if len(df_dm) else 0
    print(f"   > Diebold-Mariano/HLN: {len(df_dm)} comparações, {n_sig} significativas (Holm, α={ALPHA}) -> {out.name}")
    return df_dm


def run_friedman(metrics_dir, metric="MSE"):
    """Lê todos os *_evaluation_rolling_windows.csv e salva friedman_{summary,ranks,posthoc}.csv."""
    metrics_dir = Path(metrics_dir)
    frames = []
    for f in sorted(metrics_dir.glob("*_evaluation_rolling_windows.csv")):
        df = pd.read_csv(f, parse_dates=["cutoff"])
        df["Dataset"] = f.name.replace("_evaluation_rolling_windows.csv", "")
        frames.append(df)
    if not frames:
        print("   ⏩ Friedman: nenhum *_evaluation_rolling_windows.csv encontrado.")
        return None
    df_all = pd.concat(frames, ignore_index=True)
    blocks, block_desc = build_friedman_blocks(df_all, metric)
    summary, ranks, posthoc = friedman_tables(blocks)
    if summary.empty:
        print("   ⏩ Friedman: dados insuficientes.")
        return None
    summary["Metrica"] = metric
    summary["Bloco"] = block_desc
    n_blocks = int(summary["N_blocos"].max())
    if block_desc == "dataset" and n_blocks <= MIN_BLOCKS_RULE_OF_THUMB:
        print(f"   ⚠️ Friedman com {n_blocks} datasets: abaixo da recomendação N > {MIN_BLOCKS_RULE_OF_THUMB} (Demšar, 2006).")
    if block_desc != "dataset":
        print("   ⚠️ Friedman exploratório: só um dataset, blocos = janelas (dependentes). Use mais datasets para inferência.")
    summary.round(6).to_csv(metrics_dir / "friedman_summary.csv", index=False)
    ranks.round(4).to_csv(metrics_dir / "friedman_ranks.csv", index=False)
    posthoc.round(6).to_csv(metrics_dir / "friedman_posthoc.csv", index=False)
    print(f"   > Friedman ({metric}, bloco = {block_desc}) -> friedman_summary/ranks/posthoc.csv")
    return summary, ranks, posthoc


if __name__ == "__main__":
    import yaml

    with open(BASE_DIR / "config" / "main_config.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    run_friedman(BASE_DIR / cfg["results_paths"]["metrics"])
