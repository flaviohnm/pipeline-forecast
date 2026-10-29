import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
import numpy as np
from neuralforecast import NeuralForecast

# Define a raiz do projeto e o sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução
from src.models.deep_baselines import fit_predict_rolling, free_memory
from src.models.arima_baseline import validation_origins
from src.models.deep_profile import build_nhits, load_deep_config, model_horizons
from src.models.hybrid_variants import best_val_mse, load_hybrid_config, residual_input_sizes
from src.utils.compute import describe_compute
from src.utils.general import get_datasets, get_protocol, get_rolling_origins, prepare_data

warnings.filterwarnings("ignore")


def residual_check(nf, df_res, info, input_size, h, ljung_lag=None):
    """
    Checagem do N-HiTS dos resíduos no bloco de VALIDAÇÃO (nunca no teste).
    Pergunta: o N-HiTS prevê o resíduo futuro melhor que a previsão "zero" (que equivale ao ARIMA puro)?
    `skill = 1 - MSE(N-HiTS) / MSE(zero)`: > 0 há ganho, <= 0 o N-HiTS não acrescenta nada ao ARIMA.
    Também reporta a correlação entre o resíduo previsto e o real, a razão de desvios-padrão e o p-valor de
    Ljung-Box dos resíduos de treino (p alto = ruído branco: não há estrutura para a rede aprender).
    Retorna um dict (uma linha de {dataset}_residual_check.csv).
    """
    df = df_res.sort_values("ds").reset_index(drop=True)
    train_end = info["train_size"]
    val_end = train_end + info["rolling"]["val_size"]
    step = info["rolling"]["step_size"]

    preds, trues = [], []
    for start in range(train_end, val_end - h + 1, step):
        if start < input_size:
            continue
        context = df.iloc[start - input_size : start]
        future = nf.predict(df=context).reset_index().sort_values("ds")
        true = df["y"].iloc[start : start + h].to_numpy()
        if len(future) != h or len(true) != h or np.isnan(true).any():
            continue
        preds.append(future["NHITS"].to_numpy())
        trues.append(true)

    row = {"H": h, "janelas_validacao": len(preds)}
    if not preds:
        row.update(skill=np.nan, correlacao=np.nan, razao_desvio=np.nan, ljung_box_p=np.nan)
        return row

    pred, true = np.concatenate(preds), np.concatenate(trues)
    mse_model, mse_zero = float(np.mean((pred - true) ** 2)), float(np.mean(true**2))
    row.update(
        mse_nhits=mse_model,
        mse_zero=mse_zero,
        skill=1.0 - mse_model / mse_zero,
        correlacao=float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 and true.std() > 0 else np.nan,
        razao_desvio=float(pred.std() / true.std()) if true.std() > 0 else np.nan,
    )

    lag = ljung_lag or info.get("seasonal_period", 24)
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        resid = df["y"].iloc[:train_end].dropna().to_numpy()
        row["ljung_box_p"] = float(acorr_ljungbox(resid, lags=[lag])["lb_pvalue"].iloc[0])
    except Exception:
        row["ljung_box_p"] = np.nan
    return row


def nhits_validation_predictions(nf, df_res, input_size, h, origins):
    """
    Previsões do N-HiTS dos resíduos nas origens do bloco de VALIDAÇÃO (as mesmas do ARIMA de validação),
    para escolher o alpha. Formato longo: H, cutoff, ds, NHITS. Origem sem contexto suficiente é ignorada.
    """
    df = df_res.sort_values("ds").reset_index(drop=True)
    frames = []
    for start in origins:
        if start < input_size:
            continue
        context = df.iloc[start - input_size : start]
        future = nf.predict(df=context).reset_index().sort_values("ds")
        if len(future) < h or context["y"].isna().any():
            continue
        frame = future[["ds", "NHITS"]].iloc[:h].copy()
        frame.insert(0, "cutoff", context["ds"].iloc[-1])
        frame.insert(0, "H", h)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["H", "cutoff", "ds", "NHITS"])


def residuals_on_full_dates(df_res, df_full):
    """
    Alinha os resíduos (que terminam no último cutoff) às datas da série completa.
    As datas após o último cutoff ficam sem valor: nunca entram em treino nem em contexto
    (fit_predict_rolling falha se isso acontecer).
    """
    out = df_full[["unique_id", "ds"]].merge(df_res[["ds", "y"]], on="ds", how="left")
    return out.sort_values("ds").reset_index(drop=True)


class HybridNHITSTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos a leitura do YAML apenas para os caminhos e random_seed global
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.resid_dir = BASE_DIR / self.config["results_paths"]["residuals"]
        self.resid_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = BASE_DIR / self.config["results_paths"]["metrics"]
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = get_protocol(self.config)
        self.deep_cfg = load_deep_config(self.config)
        self.hybrid_cfg = load_hybrid_config(self.config)

    def run_rolling(self, name, info):
        """Treina o N-HiTS uma vez nos resíduos de treino+val e prevê o resíduo em cada origem do teste."""
        resid_path = self.resid_dir / f"{name}_arima_residuals_rolling.csv"
        if not resid_path.exists():
            print(f"⚠️ Resíduos rolling do ARIMA não encontrados em: {resid_path}. Pulando...")
            return

        df_res = pd.read_csv(resid_path, parse_dates=["ds"])
        df_full = prepare_data(pd.read_csv(BASE_DIR / info["path"]), info, name)
        df_res = residuals_on_full_dates(df_res, df_full)

        horizons = model_horizons(info, self.deep_cfg)
        sizes = residual_input_sizes(self.hybrid_cfg, self.deep_cfg)
        val_origins = validation_origins(info, len(df_full))
        arima_val_file = BASE_DIR / self.config["results_paths"]["arima"] / f"{name}_arima_val_rolling.csv"
        df_arima_val = pd.read_csv(arima_val_file, parse_dates=["cutoff", "ds"]) if arima_val_file.exists() else None
        if df_arima_val is None:
            print("   ⚠️ ARIMA de validação não encontrado: sem alpha nem escolha de janela por validação.")
        print(
            f"   > Rolling: estratégia {self.deep_cfg['strategy']}, H = {horizons}; treino em treino+val, "
            f"contexto de {sizes} resíduos por origem"
            + (" (escolhido na validação)" if len(sizes) > 1 else "")
        )

        out_file = self.resid_dir / f"{name}_nhits_residuals_rolling.csv"
        val_file = self.resid_dir / f"{name}_nhits_residuals_val_rolling.csv"
        per_h, per_h_val, checks = [], [], []
        for h in horizons:
            best = None
            for input_size in sizes:
                model = build_nhits(h, self.config, input_size)
                print(f"\n🚀 N-HiTS dos resíduos (rolling) H={h}, input_size={input_size}...")
                held = {}

                def after_fit(nf, h=h, input_size=input_size, held=held):
                    held["check"] = residual_check(nf, df_res, info, input_size, h)
                    held["check"]["input_size"] = input_size
                    held["val"] = nhits_validation_predictions(nf, df_res, input_size, h, val_origins)

                forecasts = fit_predict_rolling("NHITS", model, df_res, info, input_size, horizon=h, on_fitted=after_fit)
                if df_arima_val is not None and len(held["val"]):
                    score = best_val_mse(df_arima_val, held["val"], self.hybrid_cfg["alpha"]["grid_step"])
                else:
                    score = held["check"].get("mse_nhits", np.inf)
                if len(sizes) > 1:
                    print(f"   > input_size={input_size}: MSE de validação = {score:.6f}")
                if best is None or score < best[0]:
                    best = (score, input_size, forecasts, held)
                del model
                free_memory()
            _, chosen, forecasts, held = best
            forecasts.insert(1, "H", h)
            per_h.append(forecasts)
            per_h_val.append(held["val"])
            checks.append(held["check"])
            pd.concat(per_h, ignore_index=True).to_csv(out_file, index=False)
            pd.concat(per_h_val, ignore_index=True).to_csv(val_file, index=False)
        print(f"✅ Previsões rolling do N-HiTS salvas em: {self.resid_dir.name}/{out_file.name}")

        df_check = pd.DataFrame(checks)
        check_file = self.metrics_dir / f"{name}_residual_check.csv"
        df_check.round(6).to_csv(check_file, index=False)
        print(f"\n🔎 Checagem do N-HiTS nos resíduos (bloco de validação) -> {check_file.name}")
        print(df_check.round(4).to_string(index=False))
        for row in checks:
            if not np.isfinite(row.get("skill", np.nan)):
                print(f"   ⚠️ Checagem N-HiTS/resíduos H={row['H']}: sem janelas de validação suficientes.")
            elif row["skill"] <= 0:
                print(
                    f"   ⚠️ Checagem N-HiTS/resíduos H={row['H']}: sem ganho sobre prever resíduo zero "
                    f"(skill={row['skill']:+.3f}); o híbrido tende a empatar com o ARIMA."
                )

    def run(self):
        print("\n🧠 [PASSO 4] Iniciando Janelamento e Treino do N-HiTS sobre os Resíduos")
        print(describe_compute(self.config))

        # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO
        datasets_to_run = get_datasets()

        if not datasets_to_run:
            print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando.")
            return

        # 2. ITERAÇÃO LIMPA
        for name, info in datasets_to_run.items():
            print(f"\n=======================================")
            print(f"🧠 Processando N-HiTS (Resíduos) para: {name}")
            print(f"=======================================")

            if self.protocol == "rolling":
                self.run_rolling(name, info)
                continue

            # 3. Carrega os resíduos (O erro não-linear extraído do ARIMA)
            resid_path = self.resid_dir / f"{name}_arima_residuals.csv"
            
            if not resid_path.exists():
                print(f"⚠️ Arquivo de resíduos não encontrado em: {resid_path}. Pulando...")
                continue

            df_res = pd.read_csv(resid_path)
            df_res["ds"] = pd.to_datetime(df_res["ds"])

            # O horizonte de previsão máximo já vem do general.py
            horizon = info["max_horizon"]
            input_size = int(self.deep_cfg["input_size"])

            # =======================================================
            # TRAVA DE SEGURANÇA: Compatibilidade de Janela PyTorch
            # O val_size não pode ser matematicamente inferior ao horizonte
            # =======================================================
            safe_val_size = info.get("val_size", horizon)
            if safe_val_size < horizon:
                safe_val_size = horizon

            print("\n⚙️ Configuração das Janelas MIMO:")
            print(f"   > Input Window (Passado a observar): {input_size} pontos")
            print(f"   > Output Window (Futuro a prever): {horizon} pontos")
            print(f"   > Validation Size (Protegido): {safe_val_size} pontos")

            # 4. Configura a Rede Neural N-HiTS
            models = [build_nhits(horizon, self.config, input_size)]

            # 5. Inicializa o motor da Nixtla
            nf = NeuralForecast(models=models, freq=info["freq"])

            # 6. Treina a rede sobre os resíduos
            print("\n🚀 A iniciar o treino da Rede Neural...")
            nf.fit(df=df_res, val_size=safe_val_size)

            # 7. Previsão dos resíduos futuros
            print("🔮 A prever os resíduos futuros...")
            forecasts = nf.predict().reset_index()

            # 8. Salvar as predições do N-HiTS
            out_file = self.resid_dir / f"{name}_nhits_residuals.csv"
            forecasts.to_csv(out_file, index=False)
            print(f"✅ Previsões do N-HiTS salvas em: {self.resid_dir.name}/{out_file.name}")

        print("\n🏁 Processo de Treinamento N-HiTS (Resíduos) Finalizado!\n")


if __name__ == "__main__":
    trainer = HybridNHITSTrainer()
    trainer.run()