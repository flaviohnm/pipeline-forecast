#!/bin/bash

# ==============================================================================
# PREVENÇÃO DE SUSPENSÃO (Cross-Platform)
# ==============================================================================
INHIBIT_CMD=""
if command -v systemd-inhibit &> /dev/null; then
    INHIBIT_CMD="systemd-inhibit "
fi

# ==============================================================================
# LOGGING E RASTREABILIDADE
# ==============================================================================
LOG_DIR="logs"
RESULTS_DIR="results"
TIMES_CSV="${RESULTS_DIR}/execution_times.csv"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

if [ ! -f "$TIMES_CSV" ]; then
    echo "timestamp,step,duration_seconds,status,git_commit" > "$TIMES_CSV"
fi

get_git_commit() {
    git rev-parse --short HEAD 2>/dev/null || echo "N/A"
}

# ==============================================================================
# FUNÇÃO AUXILIAR
# ==============================================================================
run_and_time() {
    local step_name=$1
    local cmd=$2
    local slug
    slug=$(echo "$step_name" | tr ' /()' '_' | tr -cd '[:alnum:]_-')
    local log_file="${LOG_DIR}/${slug}_$(date +%Y%m%d_%H%M%S).log"

    echo "======================================================="
    echo "⏳ Iniciando: $step_name"
    echo "📝 Log completo em: $log_file"
    echo "======================================================="

    local start_time=$(date +%s)

    eval "$cmd" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local h=$((elapsed / 3600))
    local m=$(((elapsed % 3600) / 60))
    local s=$((elapsed % 60))

    if [ "$status" -eq 0 ]; then
        echo "-------------------------------------------------------"
        echo "⏱️  TEMPO GASTO EM [$step_name]: ${h}h ${m}m ${s}s"
        echo "-------------------------------------------------------"
    else
        echo "-------------------------------------------------------"
        echo "❌ [$step_name] FALHOU (código $status) após ${h}h ${m}m ${s}s"
        echo "   Veja o log completo em: $log_file"
        echo "-------------------------------------------------------"
    fi
    echo ""

    echo "$(date '+%Y-%m-%d %H:%M:%S'),${step_name},${elapsed},${status},$(get_git_commit)" >> "$TIMES_CSV"

    return $status
}

# ==============================================================================
# PIPELINE COMPLETO
# ==============================================================================
run_full_pipeline() {
    echo "🚀 Iniciando Execução Completa (conforme configurado em general.py)..."
    echo "🔖 Commit atual: $(get_git_commit)"
    local TOTAL_START=$(date +%s)

    run_and_time "Ingestao_de_Dados" "${INHIBIT_CMD}poetry run python src/ingestion/download_dataset.py" || return 1
    run_and_time "AutoETS_Baseline" "${INHIBIT_CMD}poetry run python src/models/ets_baseline.py" || return 1
    run_and_time "Prophet_Baseline" "${INHIBIT_CMD}poetry run python src/models/prophet_baseline.py" || return 1
    run_and_time "ARIMA_Baseline" "${INHIBIT_CMD}poetry run python src/models/arima_baseline.py" || return 1
    run_and_time "NHiTS_Treino_Residuos" "${INHIBIT_CMD}poetry run python src/models/nhits_residual.py" || return 1
    run_and_time "Hibrido_ARIMA_NHiTS" "${INHIBIT_CMD}poetry run python src/models/hybrid_combiner.py" || return 1
    run_and_time "Hibrido_Prophet_NHiTS" "${INHIBIT_CMD}poetry run python src/models/hybrid_prophet_nhits.py" || return 1
    run_and_time "Deep_Learning_Baselines" "${INHIBIT_CMD}poetry run python src/models/deep_baselines.py" || return 1
    run_and_time "Avaliacao_de_Metricas" "${INHIBIT_CMD}poetry run python src/metrics/evaluate_metrics.py" || return 1
    run_and_time "Geracao_de_Graficos" "${INHIBIT_CMD}poetry run python src/visualization/plot_results.py" || return 1

    local TOTAL_END=$(date +%s)
    local TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
    local TOTAL_H=$((TOTAL_ELAPSED / 3600))
    local TOTAL_M=$(((TOTAL_ELAPSED % 3600) / 60))
    local TOTAL_S=$((TOTAL_ELAPSED % 60))

    echo "======================================================="
    echo "🎉 Experimento Completo Finalizado com Sucesso!"
    echo "⏱️  TEMPO TOTAL DE EXECUÇÃO: ${TOTAL_H}h ${TOTAL_M}m ${TOTAL_S}s"
    echo "📂 Resultados disponíveis na pasta results/"
    echo "📊 Tempos por etapa: ${TIMES_CSV}"
    echo "======================================================="
}

# ==============================================================================
# EXECUTA UMA OPÇÃO DO MENU
# ==============================================================================
executar_opcao() {
    local OPCAO=$1
    case $OPCAO in
        0) run_and_time "Ingestao_de_Dados" "${INHIBIT_CMD}poetry run python src/ingestion/download_dataset.py" ;;
        1) run_full_pipeline ;;
        2) run_and_time "ARIMA_Baseline" "${INHIBIT_CMD}poetry run python src/models/arima_baseline.py" ;;
        3) run_and_time "AutoETS_Baseline" "${INHIBIT_CMD}poetry run python src/models/ets_baseline.py" ;;
        4) run_and_time "Prophet_Baseline" "${INHIBIT_CMD}poetry run python src/models/prophet_baseline.py" ;;
        5) run_and_time "NHiTS_Treino_Residuos" "${INHIBIT_CMD}poetry run python src/models/nhits_residual.py" ;;
        6) run_and_time "Hibrido_ARIMA_NHiTS" "${INHIBIT_CMD}poetry run python src/models/hybrid_combiner.py" ;;
        7) run_and_time "Hibrido_Prophet_NHiTS" "${INHIBIT_CMD}poetry run python src/models/hybrid_prophet_nhits.py" ;;
        8) run_and_time "Deep_Learning_Baselines" "${INHIBIT_CMD}poetry run python src/models/deep_baselines.py" ;;
        9) run_and_time "Avaliacao_de_Metricas" "${INHIBIT_CMD}poetry run python src/metrics/evaluate_metrics.py" ;;
        10) run_and_time "Geracao_de_Graficos" "${INHIBIT_CMD}poetry run python src/visualization/plot_results.py" ;;
        *)
            echo "❌ Opção inválida."
            return 1
            ;;
    esac
}

export TF_ENABLE_ONEDNN_OPTS=0

# ==============================================================================
# FORÇA UTF-8 NO PYTHON
# ==============================================================================
export PYTHONIOENCODING="utf-8"
export PYTHONUTF8=1

if ! command -v poetry &> /dev/null; then
    echo "❌ ERRO: Poetry não encontrado! Execute o bash setup_venv.sh primeiro."
    exit 1
fi

if [ -n "$1" ]; then
    executar_opcao "$1"
    exit $?
fi

# ==============================================================================
# MODO INTERATIVO
# ==============================================================================
while true; do
    clear
    echo "======================================================="
    echo "   🚀  Time Series Forecasting - Orquestrador Híbrido"
    echo "======================================================="
    echo "✅ Ambiente gerenciado pelo Poetry."
    echo "🎯 CONTROLE DE DATASETS: Gerenciado pelo arquivo src/utils/general.py"
    echo "🔖 Commit atual: $(get_git_commit)"
    echo ""
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

    if [[ "$OPCAO" == "S" || "$OPCAO" == "s" ]]; then
        echo "👋 Encerrando orquestrador. Até a próxima!"
        exit 0
    fi

    executar_opcao "$OPCAO"

    echo ""
    read -p "Pressione [ENTER] para voltar ao menu..."
done