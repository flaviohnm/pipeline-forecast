import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

# Define a raiz do projeto e o sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução
from src.utils.general import get_datasets

warnings.filterwarnings("ignore")


class HybridNHITSTrainer:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos a leitura do YAML apenas para os caminhos e random_seed global
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.resid_dir = BASE_DIR / self.config["results_paths"]["residuals"]
        self.resid_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\n🧠 [PASSO 4] Iniciando Janelamento e Treino do N-HiTS sobre os Resíduos")

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

            # 3. Carrega os resíduos (O erro não-linear extraído do ARIMA)
            resid_path = self.resid_dir / f"{name}_arima_residuals.csv"

            if not resid_path.exists():
                print(f"⚠️ Arquivo de resíduos não encontrado em: {resid_path}. Pulando...")
                continue

            df_res = pd.read_csv(resid_path)
            df_res["ds"] = pd.to_datetime(df_res["ds"])

            # O horizonte de previsão máximo já vem do general.py
            horizon = info["max_horizon"]
            input_size = horizon * 2

            print("\n⚙️ Configuração das Janelas MIMO:")
            print(f"   > Input Window (Passado a observar): {input_size} pontos")
            print(f"   > Output Window (Futuro a prever): {horizon} pontos")

            # 4. Configura a Rede Neural N-HiTS
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

            # 5. Inicializa o motor da Nixtla
            nf = NeuralForecast(models=models, freq=info["freq"])

            # 6. Treina a rede sobre os resíduos
            print("\n🚀 A iniciar o treino da Rede Neural...")
            nf.fit(df=df_res, val_size=info.get("val_size", 0))

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
