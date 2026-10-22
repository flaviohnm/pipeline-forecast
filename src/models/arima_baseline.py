import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# Define a raiz do projeto e o sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução e o preparador de dados
from src.utils.general import check_rolling_cutoffs, get_datasets, get_protocol, prepare_data, rolling_input

warnings.filterwarnings("ignore")


class ArimaScientificBaseline:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos a leitura do YAML apenas para mapear as pastas de saída (results_paths)
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["arima"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.resid_dir = BASE_DIR / self.config["results_paths"]["residuals"]
        self.resid_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = get_protocol(self.config)

    def run_rolling(self, name, info, df):
        """Protocolo rolling: ajusta uma vez até a 1ª origem e atualiza o estado nas demais (refit=False)."""
        df_cv, origins, cv_kwargs = rolling_input(df, info)
        print(
            f"   > Rolling: {len(origins)} janelas de H={cv_kwargs['h']} a cada {cv_kwargs['step_size']} passos | "
            f"ajuste em {origins[0]} pontos (treino+val), parâmetros fixos nas demais janelas"
        )

        models = [AutoARIMA(season_length=info["seasonal_period"], approximation=True)]
        sf = StatsForecast(models=models, freq=info["freq"], n_jobs=-1)
        cv = sf.cross_validation(df=df_cv, refit=False, **cv_kwargs).reset_index()
        cv = cv.rename(columns={"AutoARIMA": "ARIMA"})
        check_rolling_cutoffs(cv, df, origins)

        out_path = self.forecast_dir / f"{name}_arima_rolling.csv"
        cv[["unique_id", "cutoff", "ds", "y", "ARIMA"]].to_csv(out_path, index=False)
        print(f"   ✅ Previsões rolling salvas em: {out_path.name}")

    def run(self):
        print("\n🔬 [PASSO 1] Iniciando ARIMA Baseline Científico")
        print("Critério: Split no Horizonte Máximo")

        # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO (Substitui o .env)
        datasets_to_run = get_datasets()

        if not datasets_to_run:
            print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando.")
            return

        # 2. ITERAÇÃO LIMPA
        for name, info in datasets_to_run.items():
            max_h = info["max_horizon"]

            print(f"\n=============================================")
            print(f"📊 Processando: {name} | H_max = {max_h}")
            print(f"=============================================")

            raw_file = BASE_DIR / info["path"]
            if not raw_file.exists():
                print(f"⚠️ Arquivo bruto não encontrado para {name}: {raw_file}. Pulando...")
                continue

            df_raw = pd.read_csv(raw_file)
            df = prepare_data(df_raw, info, name)

            if self.protocol == "rolling":
                self.run_rolling(name, info, df)
                continue

            train_df = df.iloc[:-max_h]
            test_df = df.iloc[-max_h:]

            print(f"   > Treino: {len(train_df)} pontos | Teste (Blind): {len(test_df)} pontos")

            models = [AutoARIMA(season_length=info["seasonal_period"], approximation=True)]
            sf = StatsForecast(models=models, freq=info["freq"], n_jobs=-1)

            # --- MODELAGEM E PREVISÃO SIMULTÂNEA ---
            print(f"   > Ajustando ARIMA e gerando projeção recursiva para H={max_h}...")
            forecast_df = sf.forecast(df=train_df, h=max_h, fitted=True)

            # RENOMEANDO AutoARIMA para ARIMA
            forecast_df = forecast_df.rename(columns={"AutoARIMA": "ARIMA"})

            # --- SALVANDO PREVISÕES ---
            forecast_path = self.forecast_dir / f"{name}_arima_predictions.csv"
            forecast_df.to_csv(forecast_path, index=False)
            print(f"   ✅ Previsão salva em: {forecast_path.name}")

            # --- EXTRAÇÃO DE RESÍDUOS IN-SAMPLE ---
            print("   > Extraindo resíduos in-sample...")
            fitted = sf.forecast_fitted_values().reset_index()

            # Calculando o resíduo
            fitted["residual"] = fitted["y"] - fitted["AutoARIMA"]

            resid_path = self.resid_dir / f"{name}_arima_residuals.csv"
            df_res = fitted[["unique_id", "ds", "residual"]].rename(columns={"residual": "y"})
            df_res.to_csv(resid_path, index=False)
            print(f"   💾 Resíduos salvos em: {self.resid_dir.name}/{resid_path.name}")

        print("\n🏁 Passo 1 Finalizado. Resíduos prontos para a etapa Híbrida.\n")


if __name__ == "__main__":
    trainer = ArimaScientificBaseline()
    trainer.run()
