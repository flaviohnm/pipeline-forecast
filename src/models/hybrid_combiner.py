import sys
from pathlib import Path

import pandas as pd
import yaml

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Injeta a raiz do projeto no path do Python se necessário
sys.path.append(str(BASE_DIR))

# Importando o novo controlador de execução
from src.models.hybrid_variants import (
    add_alpha_column,
    alpha_by_block,
    alpha_by_horizon,
    blocks_forecast,
    load_hybrid_config,
)
from src.utils.general import get_datasets, get_protocol


def combine_rolling(df_arima, df_nhits):
    """
    Y_final = ARIMA + N-HiTS(resíduo) para cada (cutoff, ds) do protocolo rolling.
    Se df_nhits tem a coluna H (um N-HiTS por comprimento de previsão), o resultado também tem H:
    para cada H, o N-HiTS de H passos é somado às H primeiras previsões do ARIMA da janela.
    Falha se as previsões não cobrirem exatamente as mesmas janelas e datas.
    """
    keys = ["cutoff", "ds"]
    if "H" not in df_nhits.columns:
        df_final = pd.merge(
            df_arima[["unique_id"] + keys + ["ARIMA"]], df_nhits[keys + ["NHITS"]], on=keys, how="inner"
        )
        if not (len(df_final) == len(df_arima) == len(df_nhits)):
            raise ValueError(
                f"Previsões rolling desalinhadas: ARIMA={len(df_arima)} linhas, N-HiTS={len(df_nhits)}, "
                f"em comum={len(df_final)}. Rode ARIMA e N-HiTS (resíduos) com o mesmo protocolo."
            )
        df_final["Hybrid_ARIMA_NHITS"] = df_final["ARIMA"] + df_final["NHITS"]
        return df_final.sort_values(keys).reset_index(drop=True)

    arima = df_arima[["unique_id"] + keys + ["ARIMA"]]
    df_final = pd.merge(df_nhits[["H"] + keys + ["NHITS"]], arima, on=keys, how="left")
    if df_final["ARIMA"].isna().any() or set(df_nhits["cutoff"]) != set(df_arima["cutoff"]):
        raise ValueError(
            "Previsões rolling desalinhadas: o ARIMA não cobre todas as janelas/datas do N-HiTS. "
            "Rode ARIMA e N-HiTS (resíduos) com o mesmo protocolo."
        )
    counts = df_final.groupby(["H", "cutoff"]).size().reset_index(name="n")
    if (counts["n"] != counts["H"]).any():
        raise ValueError("Alguma janela do N-HiTS não tem exatamente H previsões.")
    df_final["Hybrid_ARIMA_NHITS"] = df_final["ARIMA"] + df_final["NHITS"]
    cols = ["unique_id", "H", "cutoff", "ds", "ARIMA", "NHITS", "Hybrid_ARIMA_NHITS"]
    return df_final[cols].sort_values(["H", "cutoff", "ds"]).reset_index(drop=True)


class HybridCombiner:
    def __init__(self, config_path="config/main_config.yaml"):
        # Mantemos o YAML apenas para as configurações de caminhos das pastas (results_paths)
        with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Onde buscar as previsões
        self.resid_dir = BASE_DIR / self.config["results_paths"]["residuals"]
        self.arima_dir = BASE_DIR / self.config["results_paths"]["arima"]
        self.hybrid_dir = BASE_DIR / self.config["results_paths"]["hybrid"]
        self.hybrid_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = get_protocol(self.config)
        self.hybrid_cfg = load_hybrid_config(self.config)
        paths = self.config["results_paths"]
        self.blocks_dir = BASE_DIR / paths.get("hybrid_blocks", "results/forecasts/hybrid_blocks/")
        self.metrics_dir = BASE_DIR / paths["metrics"]

    def read_validation(self, name):
        """ARIMA e N-HiTS no bloco de validação, ou None se não existirem."""
        arima_val = self.arima_dir / f"{name}_arima_val_rolling.csv"
        nhits_val = self.resid_dir / f"{name}_nhits_residuals_val_rolling.csv"
        if not (arima_val.exists() and nhits_val.exists()):
            return None
        df_a = pd.read_csv(arima_val, parse_dates=["cutoff", "ds"])
        df_n = pd.read_csv(nhits_val, parse_dates=["cutoff", "ds"])
        return (df_a, df_n) if len(df_a) and len(df_n) else None

    def run_rolling(self, name):
        arima_file = self.arima_dir / f"{name}_arima_rolling.csv"
        nhits_file = self.resid_dir / f"{name}_nhits_residuals_rolling.csv"
        for f in (arima_file, nhits_file):
            if not f.exists():
                print(f"⚠️ Arquivo rolling não encontrado para {name}: {f.name}. Pulando combinação.")
                return

        df_arima = pd.read_csv(arima_file, parse_dates=["cutoff", "ds"])
        df_nhits = pd.read_csv(nhits_file, parse_dates=["cutoff", "ds"])
        df_final = combine_rolling(df_arima, df_nhits)
        self.add_variants(name, df_final, df_nhits, df_arima)

    def add_variants(self, name, df_final, df_nhits, df_arima):
        """Escreve o híbrido por H (com a coluna alpha), o híbrido por blocos e a tabela de alphas."""
        cfg = self.hybrid_cfg
        tables, alphas_h, alphas_block = [], None, None
        val = self.read_validation(name) if (cfg["alpha"]["enabled"] or cfg["blocks"]["enabled"]) else None
        if val is None and cfg["alpha"]["enabled"]:
            print("   ⚠️ Sem previsões de validação (rode ARIMA e N-HiTS de novo): variantes com alpha puladas.")
        if val is not None and cfg["alpha"]["enabled"]:
            step = cfg["alpha"]["grid_step"]
            t_h = alpha_by_horizon(*val, step=step)
            alphas_h = dict(zip(t_h["H"], t_h["alpha"]))
            df_final = add_alpha_column(df_final, alphas_h)
            tables.append(t_h)
            if "H" in df_nhits.columns and df_nhits["H"].nunique() > 1:
                t_b = alpha_by_block(*val, step=step)
                alphas_block = dict(zip(t_b["H"], t_b["alpha"]))
                tables.append(t_b)

        out_file = self.hybrid_dir / f"{name}_hybrid_rolling.csv"
        df_final.to_csv(out_file, index=False)
        print(
            f"✅ Previsão rolling do híbrido salva em: {self.hybrid_dir.name}/{out_file.name} "
            f"({df_final['cutoff'].nunique()} janelas)"
        )

        if cfg["blocks"]["enabled"] and "H" in df_nhits.columns and df_nhits["H"].nunique() > 1:
            self.blocks_dir.mkdir(parents=True, exist_ok=True)
            df_blocks = blocks_forecast(df_arima, df_nhits, alphas_block)
            blocks_file = self.blocks_dir / f"{name}_hybrid_blocks_rolling.csv"
            df_blocks.to_csv(blocks_file, index=False)
            print(f"✅ Híbrido por blocos (DIRMO aninhado) salvo em: {blocks_file.parent.name}/{blocks_file.name}")
        elif cfg["blocks"]["enabled"]:
            print("   ℹ️ Um único H: variante por blocos coincide com o híbrido aditivo e foi pulada.")

        if tables:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            table = pd.concat(tables, ignore_index=True).round(6)
            table_file = self.metrics_dir / f"{name}_hybrid_alpha.csv"
            table.to_csv(table_file, index=False)
            print(f"🔎 Alpha escolhido na validação -> {table_file.name}")
            print(table.to_string(index=False))

    def run(self):
        print("\n🔗 [PASSO 5] Combinando as Previsões (ARIMA + N-HiTS)")
        print("Equação do Artigo: Y_final = ARIMA (Linear) + N-HiTS (Não-Linear)")

        # 1. CHAMA O NOVO CONTROLADOR DE EXECUÇÃO (Substitui o .env)
        datasets_to_run = get_datasets()

        if not datasets_to_run:
            print("❌ Nenhum dataset válido encontrado na EXECUTION_LIST. Encerrando.")
            return

        # 2. ITERAÇÃO LIMPA
        for name, info in datasets_to_run.items():
            print(f"\n=======================================")
            print(f"📊 Processando combinação para: {name}")
            print(f"=======================================")

            if self.protocol == "rolling":
                self.run_rolling(name)
                continue

            # Caminhos dos arquivos específicos deste dataset
            arima_file = self.arima_dir / f"{name}_arima_predictions.csv"
            nhits_file = self.resid_dir / f"{name}_nhits_residuals.csv"
            out_file = self.hybrid_dir / f"{name}_hybrid_predictions.csv"

            # 3. Validação robusta de caminhos
            if not arima_file.exists():
                print(f"⚠️ Previsões do ARIMA não encontradas para {name} em: {arima_file.name}. Pulando combinação.")
                continue
            if not nhits_file.exists():
                print(
                    f"⚠️ Previsões de resíduos do N-HiTS não encontradas para {name} em: {nhits_file.name}. Pulando combinação."
                )
                continue

            # 4. Carrega as Previsões do ARIMA (A Base Linear)
            df_arima = pd.read_csv(arima_file)
            df_arima["ds"] = pd.to_datetime(df_arima["ds"])

            # 5. Carrega as Previsões do N-HiTS (Os Resíduos Ajustados)
            df_nhits = pd.read_csv(nhits_file)
            df_nhits["ds"] = pd.to_datetime(df_nhits["ds"])

            # 6. Faz o Merge (Garante que estamos somando as coisas na mesma hora exata)
            df_final = pd.merge(df_arima, df_nhits, on="ds", how="inner")

            # 7. O CÁLCULO HÍBRIDO (A Mágica acontece aqui)
            if "ARIMA" in df_final.columns and "NHITS" in df_final.columns:
                df_final["Hybrid_ARIMA_NHITS"] = df_final["ARIMA"] + df_final["NHITS"]
            else:
                # Fallback caso a nomenclatura das colunas no arquivo de resíduos do N-HiTS use outra chave
                col_nhits = [
                    col for col in df_final.columns if col not in ["ds", "ARIMA", "unique_id_x", "unique_id_y"]
                ][0]
                df_final["Hybrid_ARIMA_NHITS"] = df_final["ARIMA"] + df_final[col_nhits]

            # 8. Salvar o resultado consolidado
            df_final.to_csv(out_file, index=False)

            print(f"✅ Previsão final do modelo híbrido salva em: {self.hybrid_dir.name}/{out_file.name}")
            print("-" * 50)
            print("Amostra das 5 primeiras previsões combinadas:")
            print(df_final[["ds", "ARIMA", "NHITS", "Hybrid_ARIMA_NHITS"]].head())
            print("-" * 50)

        print("\n🏁 Processo de Combinação Híbrida Finalizado.\n")


if __name__ == "__main__":
    combiner = HybridCombiner()
    combiner.run()
