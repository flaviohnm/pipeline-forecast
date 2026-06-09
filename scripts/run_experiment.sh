#!/bin/bash

# Encerra o script se houver erro crítico
set -e

# --- 1. Configuração do Ambiente ---
clear
echo "======================================================="
echo "   🚀  Time Series Forecasting - Orquestrador Híbrido"
echo "======================================================="

export TF_ENABLE_ONEDNN_OPTS=0

# Verifica se o Poetry está acessível
if ! command -v poetry &> /dev/null; then
    echo "❌ ERRO: Poetry não encontrado! Execute o setup_venv.sh primeiro."
    exit 1
fi

# --- 2. INJEÇÃO NATIVA DO .ENV ---
# Verifica se o arquivo .env existe. Se não, cria um padrão.
if [ ! -f .env ]; then
    echo "⚠️ Arquivo .env não encontrado. Criando um com DATASET=ETTh1..."
    echo "DATASET=ETTh1" > .env
fi

# Lê o arquivo .env e exporta as variáveis para a memória sem precisar de bibliotecas Python
export $(grep -v '^#' .env | xargs)

echo "✅ Ambiente gerenciado pelo Poetry."
echo "🎯 DATASET ATIVO (Lido do .env): $DATASET"
echo ""

# --- 3. Menu Interativo ---
echo "Escolha a etapa do experimento que deseja executar:"
echo ""
echo "  [0] 📥 Passo 0: Baixar Datasets (Ingestão)"
echo "  [1] 🏃 RODAR PIPELINE COMPLETO (Passos 0 a 8 sequenciais)"
echo "----------------------------------------------------------------------"
echo "  [2] 📈 Passo 1: ARIMA Baseline (Extração de Resíduos)"
echo "  [3] 📉 Passo 2: ETS Baseline (Estatístico Clássico)"
echo "  [4] 🧠 Passo 3: N-HiTS (Treinamento sobre Resíduos)"
echo "  [5] 🔗 Passo 4: Combinar Híbrido (ARIMA + N-HiTS)"
echo "  [6] 🤖 Passo 5: Deep Learning Baselines (MLP, LSTM, Informer, N-HiTS)"
echo "  [7] 📊 Passo 6: Avaliar Métricas (Benchmarking Completo)"
echo "  [8] 🎨 Passo 7: Gerar Gráficos e Visualizações"
echo "----------------------------------------------------------------------"
echo "  [9] Sair"
echo ""
read -p "Opção: " OPCAO

echo ""
case $OPCAO in
    0) poetry run python src/data/download_dataset.py ;;
    1)
        echo "🚀 Iniciando Execução Completa do Experimento para $DATASET..."
        systemd-inhibit poetry run python src/data/download_dataset.py
        systemd-inhibit poetry run python src/models/arima_baseline.py
        systemd-inhibit poetry run python src/models/ets_baseline.py
        systemd-inhibit poetry run python src/models/nhits_residual.py
        systemd-inhibit poetry run python src/models/hybrid_combiner.py
        #systemd-inhibit poetry run python src/models/deep_baselines.py
        systemd-inhibit poetry run python src/metrics/evaluate_metrics.py
        systemd-inhibit poetry run python src/visualization/plot_results.py
        echo ""
        echo "🎉 Experimento Completo Finalizado com Sucesso!"
        echo "📂 Resultados disponíveis na pasta results/"
        ;;
    2) poetry run python src/models/arima_baseline.py ;;
    3) poetry run python src/models/ets_baseline.py ;;
    4) poetry run python src/models/nhits_residual.py ;;
    5) poetry run python src/models/hybrid_combiner.py ;;
    6) poetry run python src/models/deep_baselines.py ;;
    7) poetry run python src/metrics/evaluate_metrics.py ;;
    8) poetry run python src/visualization/plot_results.py ;;
    9) exit 0 ;;
    *) echo "❌ Opção inválida." ;;
esac