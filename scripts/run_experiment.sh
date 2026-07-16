#!/bin/bash

# Encerra o script se houver erro crítico
set -e

# Loop infinito para manter o menu ativo
while true; do
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
    if [ ! -f .env ]; then
        echo "⚠️ Arquivo .env não encontrado. Criando um com DATASET=ETTh1..."
        echo "DATASET=ETTh1" > .env
    fi

    # Lê o arquivo .env e exporta as variáveis para a memória
    export $(grep -v '^#' .env | xargs)

    echo "✅ Ambiente gerenciado pelo Poetry."
    echo "🎯 DATASET ATIVO (Lido do .env): $DATASET"
    echo ""

    # --- 3. Menu Interativo ---
    echo "Escolha a etapa do experimento que deseja executar:"
    echo ""
    echo "  [0]  📥 Passo 0: Baixar Datasets (Ingestão)"
    echo "  [1]  🏃 RODAR PIPELINE COMPLETO Sequencial"
    echo "----------------------------------------------------------------------"
    echo "  [2]  📈 Passo 1: ARIMA Baseline (Estatístico/Resíduos)"
    echo "  [3]  📉 Passo 2: ETS Baseline (Estatístico Clássico)"
    echo "  [4]  🔮 Passo 3: Prophet Baseline (Estatístico Avançado)"
    echo "----------------------------------------------------------------------"
    echo "  [5]  🧠 Passo 4: N-HiTS (Treinamento sobre Resíduos ARIMA)"
    echo "  [6]  🔗 Passo 5: Combinar Híbrido (ARIMA + N-HiTS)"
    echo "  [7]  🧬 Passo 6: Híbrido Completo (Prophet + N-HiTS)"
    echo "----------------------------------------------------------------------"
    echo "  [8]  🤖 Passo 7: Deep Learning Baselines Puras (MLP, LSTM...)"
    echo "  [9]  📊 Passo 8: Avaliar Métricas (Benchmarking)"
    echo "  [10] 🎨 Passo 9: Gerar Gráficos e Visualizações"
    echo "----------------------------------------------------------------------"
    echo "  [S/s] Sair do Orquestrador"
    echo ""
    read -p "Opção: " OPCAO

    echo ""
    case $OPCAO in
        0) poetry run python src/ingestion/download_dataset.py ;;
        1)
            echo "🚀 Iniciando Execução Completa para $DATASET..."
            systemd-inhibit poetry run python src/ingestion/download_dataset.py
            systemd-inhibit poetry run python src/models/ets_baseline.py
            systemd-inhibit poetry run python src/models/prophet_baseline.py
            systemd-inhibit poetry run python src/models/arima_baseline.py
            systemd-inhibit poetry run python src/models/nhits_residual.py
            systemd-inhibit poetry run python src/models/hybrid_combiner.py
            systemd-inhibit poetry run python src/models/hybrid_prophet_nhits.py
            systemd-inhibit poetry run python src/models/deep_baselines.py
            systemd-inhibit poetry run python src/metrics/evaluate_metrics.py
            systemd-inhibit poetry run python src/visualization/plot_results.py
            echo ""
            echo "🎉 Experimento Completo Finalizado com Sucesso!"
            echo "📂 Resultados disponíveis na pasta results/"
            ;;
        2) poetry run python src/models/arima_baseline.py ;;
        3) poetry run python src/models/ets_baseline.py ;;
        4) poetry run python src/models/prophet_baseline.py ;;
        5) poetry run python src/models/nhits_residual.py ;;
        6) poetry run python src/models/hybrid_combiner.py ;;
        7) poetry run python src/models/hybrid_prophet_nhits.py ;;
        8) poetry run python src/models/deep_baselines.py ;;
        9) poetry run python src/metrics/evaluate_metrics.py ;;
        10) poetry run python src/visualization/plot_results.py ;;
        S|s) 
            echo "👋 Encerrando orquestrador. Até a próxima!"
            exit 0 
            ;;
        *) 
            echo "❌ Opção inválida." 
            ;;
    esac

    # Pausa para o usuário ler a saída do terminal antes de limpar a tela novamente
    echo ""
    read -p "Pressione [ENTER] para voltar ao menu..."
done