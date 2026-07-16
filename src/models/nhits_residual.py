import os
import warnings
from pathlib import Path

import pandas as pd
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

warnings.filterwarnings("ignore")

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class HybridNHITSTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.resid_dir = BASE_DIR / self.config["results_paths"]["residuals"]

    def run(self):
        # 1. Pega o valor do .env
        dataset_env = os.getenv("DATASET", "ETTh1").strip()

        # 2. Configura a lista de execução (Batch ou Único)
        if dataset_env.upper() == "ALL":
            print("\n🚀 Modo BATCH detectado: Executando N-HiTS (Resíduos) para TODOS os datasets.")
            datasets_to_run = list(self.config["datasets"].keys())
        else:
            datasets_to_run = [dataset_env]

        # 3. Inicia o loop de treinamento para cada dataset
        for dataset_name in datasets_to_run:
            print(f"\n🧠 [PASSO 4] Iniciando Janelamento e Treino do N-HiTS sobre os Resíduos - {dataset_name}")

            # Proteção contra chaves inválidas
            if dataset_name not in self.config["datasets"]:
                print(f"⚠️ Dataset {dataset_name} não encontrado no main_config.yaml. Pulando...")
                continue

            # Carrega as configurações específicas do dataset
            info = self.config["datasets"][dataset_name]

        # 1. Carrega os resíduos (O erro não-linear extraído do ARIMA)
        resid_path = self.resid_dir / f"{dataset_name}_arima_residuals.csv"
        if not resid_path.exists():
            raise FileNotFoundError(f"Arquivo de resíduos não encontrado em: {resid_path}")

        df_res = pd.read_csv(resid_path)
        df_res["ds"] = pd.to_datetime(df_res["ds"])

        # O horizonte de previsão máximo
        horizon = max(info["forecast_horizon"])  # 720

        input_size = horizon * 2

        print("\n⚙️ Configuração das Janelas MIMO:")
        print(f"   > Input Window (Passado a observar): {input_size} pontos")
        print(f"   > Output Window (Futuro a prever): {horizon} pontos")

        # 2. Configura a Rede Neural N-HiTS
        models = [
            NHITS(
                h=horizon,
                input_size=input_size,
                max_steps=400,  # Limite de épocas (ciclos de treino)
                batch_size=32,
                accelerator="cpu",
                early_stop_patience_steps=3,  # Para o treino se parar de melhorar
                random_seed=self.config["random_seed"],
            )
        ]

        # 3. Inicializa o motor da Nixtla
        nf = NeuralForecast(models=models, freq=info["freq"])

        # 4. Treina a rede sobre os resíduos
        print("\n🚀 A iniciar o treino da Rede Neural...")
        nf.fit(df=df_res, val_size=info["val_size"])

        # 5. Previsão dos resíduos futuros
        print("🔮 A prever os resíduos futuros...")
        forecasts = nf.predict()

        # 6. Salvar as predições do N-HiTS
        out_file = self.resid_dir / f"{dataset_name}_nhits_residuals.csv"
        forecasts.to_csv(out_file, index=False)
        print(f"✅ Previsões do híbrido salvas em: {self.resid_dir.name}/{out_file.name}\n")


if __name__ == "__main__":
    trainer = HybridNHITSTrainer()
    trainer.run()
