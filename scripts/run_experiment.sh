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
    local c
    c=$(git rev-parse --short HEAD 2>/dev/null) || { echo "N/A"; return; }
    # "+alterado": há mudanças não commitadas no código (o hash sozinho enganaria sobre o que rodou)
    if [ -n "$(git status --porcelain -- src scripts config tests 2>/dev/null)" ]; then
        echo "${c}+alterado"
    else
        echo "$c"
    fi
}

# Estado da execução atual (usado no resumo final do pipeline completo)
RUN_LOGS=()        # logs gerados nesta execução
FAILED_STEPS=""    # etapas que falharam, separadas por espaço
SKIPPED_STEPS=""   # etapas puladas por dependerem de uma etapa que falhou

# ==============================================================================
# FUNÇÃO AUXILIAR
# ==============================================================================
run_and_time() {
    local step_name=$1
    local cmd=$2
    local slug
    slug=$(echo "$step_name" | tr ' /()' '_' | tr -cd '[:alnum:]_-')
    local log_file="${LOG_DIR}/${slug}_$(date +%Y%m%d_%H%M%S).log"
    RUN_LOGS+=("$log_file")

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
# EXECUÇÃO TOLERANTE A FALHAS
# ==============================================================================
# Roda a etapa só se nenhuma dependência falhou; registra falhas sem interromper o pipeline.
# Uso: run_step "Nome_Etapa" "comando" [Dependencia1 Dependencia2 ...]
run_step() {
    local step_name=$1
    local cmd=$2
    shift 2

    local dep
    for dep in "$@"; do
        if [[ " $FAILED_STEPS $SKIPPED_STEPS " == *" $dep "* ]]; then
            echo "⏭️  Pulando [$step_name]: depende de [$dep], que falhou ou foi pulada."
            echo ""
            SKIPPED_STEPS="$SKIPPED_STEPS $step_name"
            return 1
        fi
    done

    if ! run_and_time "$step_name" "$cmd"; then
        FAILED_STEPS="$FAILED_STEPS $step_name"
        return 1
    fi
}

# Resumo de falhas e alertas da execução, no terminal e em results/run_summary_<data>.txt
write_run_summary() {
    local summary_file="${RESULTS_DIR}/run_summary_$(date +%Y%m%d_%H%M%S).txt"
    {
        echo "Resumo da execução - $(date '+%Y-%m-%d %H:%M:%S') - commit $(get_git_commit)"
        echo "======================================================="
        echo "Etapas com falha:${FAILED_STEPS:- nenhuma}"
        echo "Etapas puladas (dependência falhou):${SKIPPED_STEPS:- nenhuma}"
        echo ""
        echo "Erros e alertas encontrados nos logs (❌ erro | ⚠ aviso | 🚨 alerta de viés):"
        local found=0
        local log
        for log in "${RUN_LOGS[@]}"; do
            [ -f "$log" ] || continue
            # LC_ALL=C: busca byte a byte (o grep do Git Bash não casa emojis de 4 bytes no locale UTF-8)
            if LC_ALL=C grep -qE "❌|⚠|🚨" "$log"; then
                found=1
                echo "--- $log"
                LC_ALL=C grep -E "❌|⚠|🚨" "$log"
            fi
        done
        [ "$found" -eq 0 ] && echo "  nenhum"
    } | tee "$summary_file"
    echo ""
    echo "📝 Resumo salvo em: $summary_file"
}

# ==============================================================================
# PIPELINE COMPLETO
# ==============================================================================
# Comando Python restrito a um dataset (PF_DATASETS tem prioridade sobre a EXECUTION_LIST)
py_dataset() {
    local ds=$1
    shift
    echo "PF_DATASETS=${ds} ${INHIBIT_CMD}poetry run python $*"
}

# Todas as etapas de modelagem de UM dataset. Cada etapa/dataset roda em processo próprio:
# a falha de um dataset nunca derruba nem pula os outros; só pula quem depende dela NESTE dataset.
run_dataset_pipeline() {
    local ds=$1
    local S="_${ds}"
    echo "#######################################################"
    echo "# Dataset: ${ds}"
    echo "#######################################################"
    run_step "AutoETS_Baseline${S}" "$(py_dataset "$ds" src/models/ets_baseline.py)"
    run_step "Prophet_Baseline${S}" "$(py_dataset "$ds" src/models/prophet_baseline.py)"
    run_step "ARIMA_Baseline${S}" "$(py_dataset "$ds" src/models/arima_baseline.py)"

    # Otimização de hiperparâmetros (Optuna, hpo.enabled no YAML): um processo por rede, só no bloco de validação.
    # A rede dos resíduos precisa do ARIMA do mesmo dataset; os treinos finais só rodam se a busca do modelo concluiu.
    local m hpo_res_dep="" hpo_dep
    if [ "$HPO_ON" = "1" ]; then
        for m in MLP LSTM NHITS Informer; do
            run_step "HPO_${m}${S}" "$(py_dataset "$ds" src/models/hpo.py --models "$m")"
        done
        run_step "HPO_NHITS_RES${S}" "$(py_dataset "$ds" src/models/hpo.py --models NHITS_RES)" "ARIMA_Baseline${S}"
        hpo_res_dep="HPO_NHITS_RES${S}"
    fi

    run_step "NHiTS_Treino_Residuos${S}" "$(py_dataset "$ds" src/models/nhits_residual.py)" "ARIMA_Baseline${S}" $hpo_res_dep
    run_step "Hibrido_ARIMA_NHiTS${S}" "$(py_dataset "$ds" src/models/hybrid_combiner.py)" "ARIMA_Baseline${S}" "NHiTS_Treino_Residuos${S}"
    run_step "Hibrido_Prophet_NHiTS${S}" "$(py_dataset "$ds" src/models/hybrid_prophet_nhits.py)" $hpo_res_dep
    # Cada rede roda em um processo próprio: uma falha (ex.: falta de memória) não leva as outras junto
    for m in MLP LSTM NHITS Informer; do
        hpo_dep=""
        [ "$HPO_ON" = "1" ] && hpo_dep="HPO_${m}${S}"
        run_step "Deep_${m}${S}" "$(py_dataset "$ds" src/models/deep_baselines.py --models "$m")" $hpo_dep
    done
}

run_full_pipeline() {
    echo "🚀 Iniciando Execução Completa (conforme configurado em general.py)..."
    echo "🔖 Commit atual: $(get_git_commit)"
    local TOTAL_START=$(date +%s)

    # Sem dados não há o que treinar: a ingestão é a única etapa que interrompe o pipeline
    # (ela regenera sozinha qualquer bronze antigo com datas repetidas, buracos ou valores sentinela)
    if ! run_step "Ingestao_de_Dados" "${INHIBIT_CMD}poetry run python src/ingestion/download_dataset.py"; then
        write_run_summary
        return 1
    fi

    # Otimização de hiperparâmetros ligada? (hpo.enabled em config/main_config.yaml)
    HPO_ON=$(poetry run python src/models/hpo.py --enabled 2>/dev/null | tail -n 1)
    [ "$HPO_ON" = "1" ] || HPO_ON=0
    echo "🔧 Otimização de hiperparâmetros (Optuna): $([ "$HPO_ON" = "1" ] && echo ligada || echo desligada)"

    # Verificação prévia por dataset: só roda o que está íntegro e cabe no protocolo rolling
    local ALL_DATASETS READY="" ds
    ALL_DATASETS=$(poetry run python -m src.utils.general 2>/dev/null | tail -n 1)
    echo "📋 Datasets selecionados: ${ALL_DATASETS}"
    for ds in $ALL_DATASETS; do
        if run_step "Verificacao_Previa_${ds}" "$(py_dataset "$ds" src/ingestion/preflight.py)"; then
            READY="$READY $ds"
        else
            echo "🚫 ${ds} NÃO será executado (veja o log da verificação prévia)."
        fi
    done

    for ds in $READY; do
        run_dataset_pipeline "$ds"
    done

    # Avaliação e gráficos leem o que existir; a conferência final diz exatamente o que falta
    run_step "Avaliacao_de_Metricas" "${INHIBIT_CMD}poetry run python src/metrics/evaluate_metrics.py"
    run_step "Geracao_de_Graficos" "${INHIBIT_CMD}poetry run python src/visualization/plot_results.py" Avaliacao_de_Metricas
    run_step "Conferencia_Final" "${INHIBIT_CMD}poetry run python src/metrics/check_outputs.py" Avaliacao_de_Metricas

    local TOTAL_END=$(date +%s)
    local TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
    local TOTAL_H=$((TOTAL_ELAPSED / 3600))
    local TOTAL_M=$(((TOTAL_ELAPSED % 3600) / 60))
    local TOTAL_S=$((TOTAL_ELAPSED % 60))

    echo "======================================================="
    if [ -z "$FAILED_STEPS$SKIPPED_STEPS" ]; then
        echo "🎉 Experimento Completo Finalizado com Sucesso!"
    else
        echo "⚠️  Experimento finalizado COM FALHAS (veja o resumo abaixo)"
    fi
    echo "⏱️  TEMPO TOTAL DE EXECUÇÃO: ${TOTAL_H}h ${TOTAL_M}m ${TOTAL_S}s"
    echo "📂 Resultados disponíveis na pasta results/"
    echo "📊 Tempos por etapa: ${TIMES_CSV}"
    echo "======================================================="
    write_run_summary

    # Código de saída != 0 se algo falhou, mas só depois de rodar tudo o que era possível
    [ -z "$FAILED_STEPS$SKIPPED_STEPS" ]
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
        8)
            # Todas as redes, cada uma em um processo próprio (um modelo que falha não impede os outros)
            local m
            for m in MLP LSTM NHITS Informer; do
                run_and_time "Deep_${m}" "${INHIBIT_CMD}poetry run python src/models/deep_baselines.py --models ${m}"
            done
            ;;
        12) run_and_time "Deep_MLP" "${INHIBIT_CMD}poetry run python src/models/deep_baselines.py --models MLP" ;;
        13) run_and_time "Deep_LSTM" "${INHIBIT_CMD}poetry run python src/models/deep_baselines.py --models LSTM" ;;
        14) run_and_time "Deep_NHITS" "${INHIBIT_CMD}poetry run python src/models/deep_baselines.py --models NHITS" ;;
        15) run_and_time "Deep_Informer" "${INHIBIT_CMD}poetry run python src/models/deep_baselines.py --models Informer" ;;
        16) run_and_time "Deep_Informer_retomada" "${INHIBIT_CMD}poetry run python src/models/deep_baselines.py --models Informer --resume" ;;
        9) run_and_time "Avaliacao_de_Metricas" "${INHIBIT_CMD}poetry run python src/metrics/evaluate_metrics.py" ;;
        10) run_and_time "Geracao_de_Graficos" "${INHIBIT_CMD}poetry run python src/visualization/plot_results.py" ;;
        11) run_and_time "Diagnostico_GPU_CPU" "poetry run python src/utils/compute.py" ;;
        17) run_and_time "Verificacao_Previa" "poetry run python src/ingestion/preflight.py" ;;
        18) run_and_time "Conferencia_Final" "poetry run python src/metrics/check_outputs.py" ;;
        19)
            # Busca de hiperparâmetros de todas as redes (rede dos resíduos exige o ARIMA já rodado), um processo por modelo
            local k
            for k in MLP LSTM NHITS Informer NHITS_RES; do
                run_and_time "HPO_${k}" "${INHIBIT_CMD}poetry run python src/models/hpo.py --models ${k}"
            done
            ;;
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
# Não usar PYTHONUTF8=1: no Windows em português, o cmdstanpy lê a saída do where.exe (cp850)
# como UTF-8, falha e o Prophet fica sem backend ('Prophet' object has no attribute 'stan_backend').
# PYTHONIOENCODING já garante os emojis nos logs.

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
    echo "  [8]  🤖 Passo 7: Deep Learning Baselines Puras (todas, uma por processo)"
    echo "  [12] 🤖   só MLP   | [13] só LSTM | [14] só N-HiTS | [15] só Informer"
    echo "  [16] ♻️    Retomar Informer (pula os H já concluídos)"
    echo "  [9]  📊 Passo 8: Avaliar Métricas (Benchmarking)"
    echo "  [10] 🎨 Passo 9: Gerar Gráficos e Visualizações"
    echo "  [11] 🖥️  Diagnóstico: GPU ou CPU? (dispositivo que as redes vão usar)"
    echo "  [17] 🔎 Verificação prévia dos dados   | [18] ✅ Conferência final (falta algum modelo?)"
    echo "  [19] 🔧 Otimização de hiperparâmetros das redes (Optuna)"
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