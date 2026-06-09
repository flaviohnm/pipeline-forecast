from pathlib import Path

import pandas as pd
import yaml
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class ArimaScientificBaseline:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["arima"]
        self.resid_dir = BASE_DIR / self.config["results_paths"]["statistical"]

        # Garante a criação das pastas
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.resid_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, df, info, name):
        """Formatação padrão Nixtla"""
        df_n = df[[info["date_column"], info["target_column"]]].copy()
        df_n.columns = ["ds", "y"]
        df_n["unique_id"] = name
        df_n["ds"] = pd.to_datetime(df_n["ds"])
        return df_n

    def run(self):
        print("\n🔬 [PASSO 1] Iniciando ARIMA Baseline Científico")
        print("Critério: Split em T - 720 (Horizonte Máximo)")

        for name, info in self.config["datasets"].items():
            # O horizonte máximo (720 no ETT) define o isolamento do teste
            # Para o AirPassengers usamos o valor definido no YAML (24)
            max_h = info.get("max_horizon", max(info["forecast_horizon"]))

            print(f"\n📊 Processando: {name} | H_max = {max_h}")

            # 1. Carga de Dados
            df_raw = pd.read_csv(BASE_DIR / info["path"])
            df = self.prepare_data(df_raw, info, name)

            # 2. Split Rigoroso: Treino termina antes do horizonte de teste
            train_df = df.iloc[:-max_h]
            test_df = df.iloc[-max_h:]

            print(f"   > Treino: {len(train_df)} pontos | Teste (Blind): {len(test_df)} pontos")

            # 3. Configuração do Modelo (Aproximação para velocidade no i7)
            models = [AutoARIMA(season_length=info["seasonal_period"], approximation=True)]
            sf = StatsForecast(models=models, freq=info["freq"], n_jobs=-1)

            # 4. Treinamento (Fit) apenas no treino
            sf.fit(df=train_df)

            # 5. Extração de Resíduos In-Sample (O que a rede neural vai aprender)
            # Esses resíduos são a base do janelamento do Passo 2
            fitted = sf.forecast_fitted_values()
            fitted["residual"] = fitted["y"] - fitted["AutoARIMA"]

            resid_path = self.resid_dir / f"{name}_residuals.csv"
            df_res = fitted[["unique_id", "ds", "residual"]].rename(columns={"residual": "y"})
            df_res.to_csv(resid_path, index=False)
            print(f"   💾 Resíduos in-sample salvos em: {resid_path.name}")

            # 6. Previsão Out-of-Sample (Recursiva)
            print(f"   🔮 Gerando projeção recursiva para os próximos {max_h} pontos...")
            forecast_df = sf.predict(h=max_h)

            forecast_path = self.forecast_dir / f"{name}_predictions.csv"
            forecast_df.to_csv(forecast_path, index=False)
            print(f"   ✅ Previsão concluída: {forecast_path.name}")

        print("\n🏁 Passo 1 Finalizado. Resíduos prontos para a etapa Híbrida.\n")


if __name__ == "__main__":
    trainer = ArimaScientificBaseline()
    trainer.run()
