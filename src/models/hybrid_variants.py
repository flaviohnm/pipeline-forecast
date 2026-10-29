"""
Variantes do híbrido ARIMA + N-HiTS (protocolo rolling).

Todas parten da mesma previsão ARIMA e somam a ela uma correção do N-HiTS treinado nos resíduos:

1. Hybrid_ARIMA_NHITS             : ARIMA + N-HiTS(H)               (aditivo MIMO, um modelo por H)
2. Hybrid_ARIMA_NHITS_alpha       : ARIMA + alpha(H) * N-HiTS(H)    (alpha escolhido na VALIDAÇÃO; 0 = ARIMA puro)
3. Hybrid_ARIMA_NHITS_blocks      : ARIMA + correção por blocos de passos (estilo DIRMO, Ben Taieb et al., 2012)
4. Hybrid_ARIMA_NHITS_blocks_alpha: ARIMA + alpha(bloco) * correção por blocos

Blocos: com os modelos H = [24, 48, 168, ...], o passo i da janela vem do N-HiTS de menor H_k >= i, isto é,
bloco k = passos (H_{k-1}, H_k]. Não treina nada novo: reaproveita os modelos MIMO já treinados.
Isto é um DIRMO aninhado, não o DIRMO original (que treinaria uma rede por bloco com saída apenas nos passos
do bloco); o NeuralForecast só gera previsões contíguas a partir de t+1, então o original não é viável.

Alpha: minimiza o MSE de (y - ARIMA - alpha * N-HiTS) nas janelas do bloco de validação, em grade [0, 1].
Ressalva: o N-HiTS usa esse bloco para early stopping, então suas previsões na validação são otimistas
e alpha tende a ficar um pouco maior do que seria com dados inéditos.
"""

import numpy as np
import pandas as pd

KEYS = ["cutoff", "ds"]
ALPHA_COL = "Hybrid_ARIMA_NHITS_alpha"
BLOCKS_COL = "Hybrid_ARIMA_NHITS_blocks"
BLOCKS_ALPHA_COL = "Hybrid_ARIMA_NHITS_blocks_alpha"

DEFAULT_HYBRID_CONFIG = {
    "residual_input_size": None,  # None -> usa deep_learning.input_size
    "residual_input_size_grid": None,  # lista: escolhe na validação (HPO leve); tem prioridade sobre o valor único
    "alpha": {"enabled": True, "grid_step": 0.05},
    "blocks": {"enabled": True},
}


def load_hybrid_config(config):
    """Seção `hybrid` do YAML com padrões; valida os valores."""
    user = config.get("hybrid") or {}
    cfg = {
        "residual_input_size": user.get("residual_input_size", DEFAULT_HYBRID_CONFIG["residual_input_size"]),
        "residual_input_size_grid": user.get("residual_input_size_grid"),
        "alpha": {**DEFAULT_HYBRID_CONFIG["alpha"], **(user.get("alpha") or {})},
        "blocks": {**DEFAULT_HYBRID_CONFIG["blocks"], **(user.get("blocks") or {})},
    }
    step = cfg["alpha"]["grid_step"]
    if not 0 < float(step) <= 1:
        raise ValueError(f"hybrid.alpha.grid_step deve estar em (0, 1], recebido {step}.")
    grid = cfg["residual_input_size_grid"]
    if grid is not None and (not isinstance(grid, (list, tuple)) or not grid or any(int(s) <= 0 for s in grid)):
        raise ValueError("hybrid.residual_input_size_grid deve ser uma lista de inteiros positivos.")
    return cfg


def residual_input_sizes(hybrid_cfg, deep_cfg):
    """Tamanhos de janela candidatos para o N-HiTS dos resíduos (1 = sem HPO)."""
    if hybrid_cfg["residual_input_size_grid"]:
        return sorted({int(s) for s in hybrid_cfg["residual_input_size_grid"]})
    size = hybrid_cfg["residual_input_size"] or deep_cfg["input_size"]
    return [int(size)]


def alpha_grid(step):
    n = int(round(1.0 / float(step)))
    return np.linspace(0.0, 1.0, n + 1)


def add_step(df, group_cols):
    """Número do passo (1..H) de cada linha dentro do grupo, em ordem de data."""
    df = df.sort_values(group_cols + ["ds"]).reset_index(drop=True)
    df["step"] = df.groupby(group_cols).cumcount() + 1
    return df


def pick_alpha(err, corr, grid):
    """
    err = y - ARIMA, corr = previsão do N-HiTS. Retorna (alpha, mse com alpha, mse com alpha=0, mse com alpha=1).
    Empates ficam com o menor alpha (o mais conservador, mais perto do ARIMA).
    """
    err, corr = np.asarray(err, float), np.asarray(corr, float)
    mses = np.array([np.mean((err - a * corr) ** 2) for a in grid])
    best = int(np.argmin(mses))
    return float(grid[best]), float(mses[best]), float(np.mean(err**2)), float(np.mean((err - corr) ** 2))


def _validation_frame(df_arima_val, df_nhits_val):
    """Junta ARIMA (y, previsão) e N-HiTS nas mesmas janelas/datas da validação."""
    val = pd.merge(
        df_nhits_val[["H"] + KEYS + ["NHITS"]], df_arima_val[KEYS + ["y", "ARIMA"]], on=KEYS, how="inner"
    )
    if val.empty:
        raise ValueError("Validação sem janelas em comum entre ARIMA e N-HiTS (confira os arquivos *_val_*).")
    n_expected = df_nhits_val.groupby("H").size()
    n_got = val.groupby("H").size()
    if not n_expected.equals(n_got.reindex(n_expected.index)):
        raise ValueError("Previsões do ARIMA na validação não cobrem todas as datas do N-HiTS.")
    val["err"] = val["y"] - val["ARIMA"]
    return add_step(val, ["H", "cutoff"])


def block_edges(horizons):
    """{H_k: (início exclusivo, fim inclusivo]} dos blocos aninhados."""
    hs = sorted(int(h) for h in horizons)
    return {h: (prev, h) for prev, h in zip([0] + hs[:-1], hs)}


def alpha_by_horizon(df_arima_val, df_nhits_val, step=0.05):
    """alpha por H, usando todos os H passos do N-HiTS de cada H. Retorna DataFrame (uma linha por H)."""
    val, grid, rows = _validation_frame(df_arima_val, df_nhits_val), alpha_grid(step), []
    for h, sub in val.groupby("H"):
        a, m, m0, m1 = pick_alpha(sub["err"], sub["NHITS"], grid)
        rows.append(
            {"variante": "alpha_H", "H": int(h), "bloco_inicio": 1, "bloco_fim": int(h), "alpha": a,
             "mse_val": m, "mse_val_arima": m0, "mse_val_aditivo": m1, "n_janelas_val": sub["cutoff"].nunique()}
        )
    return pd.DataFrame(rows)


def alpha_by_block(df_arima_val, df_nhits_val, step=0.05):
    """alpha por bloco de passos, usando só os passos do bloco no N-HiTS de H = fim do bloco."""
    val = _validation_frame(df_arima_val, df_nhits_val)
    grid, rows = alpha_grid(step), []
    for h, (lo, hi) in block_edges(val["H"].unique()).items():
        sub = val[(val["H"] == h) & (val["step"] > lo) & (val["step"] <= hi)]
        if sub.empty:
            continue
        a, m, m0, m1 = pick_alpha(sub["err"], sub["NHITS"], grid)
        rows.append(
            {"variante": "alpha_bloco", "H": int(h), "bloco_inicio": lo + 1, "bloco_fim": hi, "alpha": a,
             "mse_val": m, "mse_val_arima": m0, "mse_val_aditivo": m1, "n_janelas_val": sub["cutoff"].nunique()}
        )
    return pd.DataFrame(rows)


def compose_blocks(df_nhits):
    """
    Correção por blocos: para cada janela, o passo i vem do N-HiTS de menor H >= i.
    Retorna cutoff, ds, step, H (modelo de origem), NHITS_blocks. Uma linha por (cutoff, ds).
    """
    df = add_step(df_nhits, ["H", "cutoff"])
    parts = []
    for h, (lo, hi) in block_edges(df["H"].unique()).items():
        sub = df[(df["H"] == h) & (df["step"] > lo) & (df["step"] <= hi)]
        parts.append(sub[["cutoff", "ds", "step", "H", "NHITS"]])
    out = pd.concat(parts, ignore_index=True).rename(columns={"NHITS": "NHITS_blocks"})
    if out.duplicated(KEYS).any():
        raise ValueError("Composição por blocos gerou datas duplicadas numa janela.")
    return out.sort_values(KEYS).reset_index(drop=True)


def blocks_forecast(df_arima, df_nhits, alpha_blocks=None):
    """
    Previsão por blocos para cada (cutoff, ds) coberta pelo maior H: ARIMA + correção do bloco
    e, se `alpha_blocks` ({H do bloco: alpha}) for dado, ARIMA + alpha(bloco) * correção.
    """
    comp = compose_blocks(df_nhits)
    out = pd.merge(comp, df_arima[["unique_id"] + KEYS + ["ARIMA"]], on=KEYS, how="left")
    if out["ARIMA"].isna().any():
        raise ValueError("O ARIMA não cobre todas as datas da composição por blocos.")
    out[BLOCKS_COL] = out["ARIMA"] + out["NHITS_blocks"]
    if alpha_blocks is not None:
        out[BLOCKS_ALPHA_COL] = out["ARIMA"] + out["H"].map(alpha_blocks).astype(float) * out["NHITS_blocks"]
    cols = ["unique_id", "cutoff", "ds", "step", "H", "ARIMA", "NHITS_blocks", BLOCKS_COL]
    cols += [BLOCKS_ALPHA_COL] if alpha_blocks is not None else []
    # "H_bloco" (e não "H"): o arquivo tem um único modelo composto por janela, sem escolha por horizonte
    return out[cols].rename(columns={"H": "H_bloco"})


def add_alpha_column(df_hybrid, alphas_h):
    """Acrescenta ARIMA + alpha(H) * N-HiTS ao DataFrame do híbrido por H. `alphas_h`: {H: alpha}."""
    missing = set(df_hybrid["H"].unique()) - set(alphas_h)
    if missing:
        raise ValueError(f"Sem alpha para H = {sorted(missing)}.")
    out = df_hybrid.copy()
    out[ALPHA_COL] = out["ARIMA"] + out["H"].map(alphas_h).astype(float) * out["NHITS"]
    return out


def best_val_mse(df_arima_val, df_nhits_val_h, step=0.05):
    """Menor MSE de validação do híbrido com alpha em grade (critério para escolher o input_size)."""
    return float(alpha_by_horizon(df_arima_val, df_nhits_val_h, step)["mse_val"].iloc[0])
