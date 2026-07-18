import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from prophet import Prophet

# Define a raiz do projeto e ajusta o sys.path para importar a pasta src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução e preparador
from src.utils.general import get_datasets, prepare_data

warnings.filterwarnings("ignore")


class ProphetBaselineTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos o YAML apenas para as configurações de caminhos das pastas (results_paths)
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.forecast_dir = BASE_DIR / self.config["results_paths"]["prophet"]
        self.forecast_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\n🔮 Iniciando Treinamento: Baseline Estatística (Prophet)")

        # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO
        datasets_to_run = get_datasets()

        if not datasets_to_run:
            print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando.")
            return

        # 2. ITERAÇÃO LIMPA
        for name, info in datasets_to_run.items():
            print(f"\n=======================================")
            print(f"🔮 Processando Prophet para: {name}")
            print(f"=======================================")

            raw_file = BASE_DIR / info["path"]
            if not raw_file.exists():
                print(f"⚠️ Arquivo bruto não encontrado para {name}: {raw_file}. Pulando...")
                continue

            df_raw = pd.read_csv(raw_file)
            df = prepare_data(df_raw, info, name)

            # Usando o max_horizon que já vem configurado do general.py
            horizon = info["max_horizon"]
            train_df = df.iloc[:-horizon]

            print("🚀 Treinando o modelo Prophet...")
            # O Prophet descobre a sazonalidade automaticamente por padrão
            model = Prophet()
            model.fit(train_df)

            print(f"🔮 Prevendo os próximos {horizon} pontos com Prophet...")
            # Cria um dataframe apenas com as datas futuras
            future = model.make_future_dataframe(periods=horizon, freq=info["freq"], include_history=False)
            forecast = model.predict(future)

            # Formata a saída para o padrão do nosso avaliador
            final_forecast = forecast[["ds", "yhat"]].rename(columns={"yhat": "Prophet"})
            final_forecast.insert(0, "unique_id", name)

            out_file = self.forecast_dir / f"{name}_prophet_predictions.csv"
            final_forecast.to_csv(out_file, index=False)
            print(f"✅ Previsão Prophet salva em: {self.forecast_dir.name}/{out_file.name}")

        print("\n🏁 Processo de Baselines do Prophet Finalizado!\n")


if __name__ == "__main__":
    trainer = ProphetBaselineTrainer()
    trainer.run()
