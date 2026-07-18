import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

# Define a raiz do projeto e o sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução e o preparador de dados
from src.utils.general import get_datasets, prepare_data

warnings.filterwarnings("ignore")

class ETSBaselineTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos a leitura do YAML apenas para mapear as pastas de saída (results_paths)
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["ets"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\n📉 Iniciando Treinamento: Baseline Estatística (AutoETS)")

        # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO
        datasets_to_run = get_datasets()

        if not datasets_to_run:
            print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando.")
            return

        # 2. ITERAÇÃO LIMPA DIRETAMENTE SOBRE O DICIONÁRIO
        for name, info in datasets_to_run.items():
            print(f"\n=============================================")
            print(f"📈 Processando AutoETS: {name}")
            print(f"=======================================")

            raw_file = BASE_DIR / info["path"]
            if not raw_file.exists():
                print(f"⚠️ Arquivo bruto não encontrado para {name}: {raw_file}. Pulando...")
                continue
                
            df_raw = pd.read_csv(raw_file)
            df = prepare_data(df_raw, info, name)

            # Os parâmetros já vêm mastigados do general.py
            horizon = info["max_horizon"]
            train_df = df.iloc[:-horizon]

            season_length = info["seasonal_period"]

            models = [AutoETS(season_length=season_length, model="ZZA")]

            sf = StatsForecast(df=train_df, models=models, freq=info["freq"], n_jobs=-1)

            print(f"🚀 Calculando AutoETS para {name} (Sazonalidade: {season_length})...")
            forecasts = sf.forecast(h=horizon)

            forecasts = forecasts.reset_index().rename(columns={"AutoETS": "ETS"})

            out_file = self.forecast_dir / f"{name}_ets_predictions.csv"
            forecasts.to_csv(out_file, index=False)
            print(f"✅ Previsão ETS salva em: {self.forecast_dir.name}/{out_file.name}")


if __name__ == "__main__":
    trainer = ETSBaselineTrainer()
    trainer.run()