# 🚀 Pipeline MLOps para Time Series Forecasting (Híbrido)

Este repositório contém o código-fonte e o pipeline MLOps desenvolvido para avaliar modelos estatísticos, de **Deep Learning** e arquiteturas híbridas na previsão univariada de séries temporais (*Time Series Forecasting*).

Embora os benchmarks ETT, Weather e ECL sejam originalmente multivariados, o experimento é **univariado**: cada modelo usa apenas a série histórica da variável-alvo, sem covariáveis. Nos datasets ETT, o alvo é a coluna `OT` (*Oil Temperature*, temperatura do óleo do transformador); em Weather e ECL, é a coluna `OT` definida como alvo nesses benchmarks; em AirPassengers, é `Passengers`. As demais colunas são descartadas na ingestão (`raw → bronze`).

O projeto automatiza todo o ciclo de vida do experimento, desde a ingestão de dados de benchmarks renomados (ETT, Weather, ECL e AirPassengers), passando pelo treinamento dos modelos, extração de resíduos, inferência e avaliação multi-horizonte.

O objetivo da pesquisa é comparar o desempenho dos modelos híbridos desenvolvidos contra o melhor modelo individual de cada conjunto de dados.

---

# 🎯 Arquitetura e Modelos

A proposta metodológica utiliza modelagem estatística para capturar os componentes lineares da série temporal e redes neurais profundas para modelar os resíduos não lineares.

## Modelos Estatísticos

- AutoARIMA
- AutoETS
- Prophet

## Modelos de Deep Learning

- N-HiTS
- Informer
- LSTM
- MLP

## Modelos Híbridos (Contribuição da Pesquisa)

- ARIMA + N-HiTS
- Prophet + N-HiTS

---

# ⚙️ Pré-requisitos

Para garantir a reprodutibilidade dos experimentos, recomenda-se:

- Sistema Operacional
  - Linux (Ubuntu/Fedora)
  - macOS
  - Windows (via Git Bash / pyenv-win)

- Python
  - **3.11.9** ou superior

- Gerenciador de Dependências
  - Poetry

---

# 🛠️ Instalação

## 1. Conceda permissão aos scripts

```bash
chmod +x scripts/setup_venv.sh scripts/run_experiment.sh
```

## 2. Crie o ambiente virtual

```bash
./scripts/setup_venv.sh
```

O script realiza automaticamente:

- Verificação da versão do Python;
- Instalação do Poetry (caso necessário);
- Criação do ambiente `.venv`;
- Instalação de todas as dependências do projeto.

---

# ⚙️ Configuração

A seleção de datasets é feita através da lista `EXECUTION_LIST`, em `src/utils/general.py`.

Exemplo:

```python
# Opções individuais: "AirPassengers", "ETTh1", "ETTh2", "ETTm1", "ECL", "Weather"
# Para rodar todos de uma vez: ["ALL"]
EXECUTION_LIST = ["ETTh1"]
```

Os caminhos usados pelo pipeline (dados e resultados) ficam centralizados em `config/main_config.yaml`:

```yaml
data_paths:
  raw: "data/raw/"      # datasets exatamente como baixados da fonte
  bronze: "data/bronze/"  # datasets já filtrados (date_column + target_column)

results_paths:
  forecasts: "results/forecasts/"
  metrics: "results/metrics/"
  # ...
```

## Protocolo de avaliação

Também em `config/main_config.yaml`:

```yaml
evaluation:
  norm_stats: "global"      # global | train  -> média/desvio da normalização z-score das métricas
  protocol: "single_split"  # single_split | rolling
```

| Protocolo | Treino | Teste | Saídas de métricas |
| --- | --- | --- | --- |
| `single_split` (padrão, original) | série inteira menos os últimos `max_horizon` pontos | uma janela: os últimos `max_horizon` pontos | `{dataset}_evaluation_complete.csv` |
| `rolling` | blocos treino/val/teste de 12/4/4 meses (ETT); cada modelo é treinado uma vez em treino+val | várias origens no bloco de teste, a cada 1 semana (13 janelas de 720h no ETTh1) | `{dataset}_evaluation_rolling.csv` (média ± desvio entre janelas) e `{dataset}_evaluation_rolling_windows.csv` (erro por janela) |

Os blocos do `rolling` ficam em `src/utils/general.py` (`train_size` e `rolling` de cada dataset; hoje só ETTh1, ETTh2 e ETTm1).
No `rolling`, cada modelo grava `{dataset}_{modelo}_rolling.csv` (formato longo: `cutoff, ds, previsão`):

- **ARIMA / ETS:** ajuste único em treino+val; nas demais origens os parâmetros ficam fixos e só o estado é atualizado.
- **Prophet:** reajustado em cada origem com os dados até o cutoff.
- **Redes (MLP, LSTM, N-HiTS, Informer):** um modelo MIMO por comprimento de previsão H (24, 48, 168, 336, 720), cada um treinado uma vez em treino+val com early stopping no bloco de validação; cada origem é prevista a partir do contexto anterior ao cutoff. Saída com coluna `H`.
- **Híbrido ARIMA + N-HiTS:** resíduos de 1 passo do ARIMA com parâmetros fixos; um N-HiTS por H treinado nos resíduos de treino+val. Antes do teste, `results/metrics/{dataset}_residual_check.csv` mede (no bloco de validação) se o N-HiTS prevê o resíduo melhor que "zero" (`skill > 0`); `skill <= 0` gera alerta ⚠️, pois o híbrido tende a empatar com o ARIMA.
- **Variantes do híbrido ARIMA + N-HiTS** (seção `hybrid` do config; ver `src/models/hybrid_variants.py`), todas avaliadas no mesmo protocolo e nas mesmas janelas:
  1. `Hybrid_ARIMA_NHITS` – aditivo: ARIMA + N-HiTS(H);
  2. `Hybrid_ARIMA_NHITS_alpha` – ARIMA + α(H)·N-HiTS(H), com α ∈ [0, 1] escolhido por H no bloco de **validação** (α = 0 reproduz o ARIMA);
  3. `Hybrid_ARIMA_NHITS_blocks` – correção por blocos de passos (DIRMO aninhado, Ben Taieb et al., 2012): o passo i vem do N-HiTS de menor H ≥ i; arquivo em `results/forecasts/hybrid_blocks/`;
  4. `Hybrid_ARIMA_NHITS_blocks_alpha` – blocos com α por bloco.
  O ARIMA de validação (ajustado só no treino) fica em `results/forecasts/arima/{dataset}_arima_val_rolling.csv`, as previsões do N-HiTS na validação em `results/residuals/{dataset}_nhits_residuals_val_rolling.csv` e os α escolhidos em `results/metrics/{dataset}_hybrid_alpha.csv`. `hybrid.residual_input_size` (padrão 96, como Liu et al., 2025) define o contexto do N-HiTS dos resíduos; `hybrid.residual_input_size_grid: [96, 336]` escolhe o contexto por H na validação (HPO leve; nunca no teste).
  Ressalvas: o N-HiTS usa o bloco de validação para early stopping, então suas previsões ali são otimistas e α tende a ser um pouco maior do que seria com dados inéditos; o DIRMO aninhado não é o DIRMO original (uma rede por bloco), que o NeuralForecast não permite; exige rodar ARIMA (passo 1) e N-HiTS dos resíduos (passo 5) de novo para gerar os arquivos de validação.
- **Híbrido Prophet + N-HiTS:** um N-HiTS por H treinado nos resíduos do Prophet de treino+val; em cada origem o Prophet é reajustado e os resíduos do contexto recalculados.
- **Ingênuos (último valor e sazonal):** calculados na própria avaliação, nos dois protocolos.

Recomendado para a dissertação: `protocol: "rolling"` com `norm_stats: "train"` (já é o padrão do `main_config.yaml`).

### Redes neurais: hiperparâmetros e estratégia por horizonte

As seções `deep_learning` e `compute` de `config/main_config.yaml` controlam todas as redes (e o N-HiTS dos híbridos):

```yaml
compute:
  accelerator: "auto"      # auto | cpu | gpu
  devices: 1
deep_learning:
  strategy: "per_horizon"  # per_horizon | single_truncated
  input_size: 336          # lookback único (como PatchTST/DLinear no ETTh1)
  common: {loss: "mse", scaler_type: "standard", max_steps: 1000, val_check_steps: 100, early_stop_patience_steps: 5}
  models: {MLP: {}, LSTM: {...}, NHITS: {...}, Informer: {...}}   # sobrescritas por modelo
```

- `per_horizon` (padrão): uma rede MIMO por H, como Informer, PatchTST e DLinear. **Não** é a estratégia Direct de Ben Taieb et al. (2012), que treina um modelo por passo. `single_truncated`: uma única rede com H = `max_horizon`, usando os primeiros h passos (comportamento anterior).
- Valores padrão e origem de cada um (Informer: d_model 512, 8 cabeças, lr 1e-4; N-HiTS: lr 1e-3; perda MSE) estão comentados em `src/models/deep_profile.py`.
- Treinar uma rede por H aumenta o custo: cinco horizontes × quatro redes (mais os N-HiTS dos híbridos).

## GPU (Ubuntu) ou CPU

O mesmo código roda em máquinas com e sem GPU. Com `compute.accelerator: "auto"` as redes usam a GPU CUDA quando o PyTorch a enxerga e a CPU caso contrário; `cpu` força CPU e `gpu` exige GPU (sem ela, cai para CPU com aviso ⚠️). O dispositivo escolhido é impresso no início de cada treino.

Checklist para Ubuntu com GPU NVIDIA:

1. `nvidia-smi` deve funcionar. O `poetry.lock` fixa PyTorch 2.12 com CUDA 13, que em geral exige driver NVIDIA recente (580 ou superior); com driver mais antigo o pipeline roda, mas na CPU.
2. `chmod +x scripts/*.sh && ./scripts/setup_venv.sh` (ou `poetry install`).
3. No menu de `scripts/run_experiment.sh`, a opção **11** mostra qual dispositivo será usado e o motivo (equivale a `poetry run python src/utils/compute.py`).
4. Statsforecast (ARIMA, ETS), Prophet e as métricas continuam na CPU.

Os resultados em GPU e CPU não são idênticos bit a bit (operações CUDA não determinísticas); registre o dispositivo usado junto dos resultados.

### Testes estatísticos

Rodar a etapa 9 (avaliação) no protocolo `rolling` também gera, em `results/metrics/`:

| Arquivo | Conteúdo |
| --- | --- |
| `{dataset}_dm_tests.csv` | Diebold-Mariano com correção de Harvey-Leybourne-Newbold, par a par, por horizonte e métrica (MSE e MAE), com p-valor ajustado por Holm. A perda de cada janela é comparada; janelas sobrepostas entram como autocorrelação (`lags = ceil(h/passo) - 1`). `diferenca_media = perda(A) - perda(B)`; positiva significa que A errou mais. |
| `friedman_summary.csv`, `friedman_ranks.csv`, `friedman_posthoc.csv` | Friedman + Iman-Davenport, ranks médios, diferença crítica de Nemenyi e pós-teste de Holm. Com **um único dataset**, os blocos são as janelas (dependentes) e o resultado é só exploratório; com vários datasets, o bloco é o dataset (Demšar recomenda N > 10). |

O módulo é `src/metrics/statistical_tests.py` e pode ser rodado isoladamente (`python src/metrics/statistical_tests.py`) para refazer só o Friedman. Referências: Ben Taieb et al. (2012), Demšar (2006), García & Herrera (2009), Diebold & Mariano (1995), Harvey, Leybourne & Newbold (1997).


---

# 🚀 Executando o Pipeline

Execute o orquestrador:

```bash
./scripts/run_experiment.sh
```

Será exibido o seguinte menu:

```text
=======================================================
🚀 Time Series Forecasting - Orquestrador Híbrido
=======================================================

✅ Ambiente gerenciado pelo Poetry.
🎯 CONTROLE DE DATASETS: Gerenciado pelo arquivo src/utils/general.py
🔖 Commit atual: a1b2c3d

Escolha a etapa do experimento:

[0]  📥 Baixar Datasets
[1]  🏃 Executar Pipeline Completo

--------------------------------------------------------

[2]  📈 ARIMA Baseline
[3]  📉 ETS Baseline
[4]  🔮 Prophet Baseline

--------------------------------------------------------

[5]  🧠 N-HiTS sobre resíduos ARIMA
[6]  🔗 Híbrido ARIMA + N-HiTS
[7]  🧬 Híbrido Prophet + N-HiTS

--------------------------------------------------------

[8]  🤖 Deep Learning Baselines
[9]  📊 Avaliação de Métricas
[10] 🎨 Geração de Gráficos

--------------------------------------------------------

[S] Sair
```

## Execução não-interativa (background / servidor remoto)

Para rodar uma etapa direto, sem menu — útil junto com `nohup`, `tmux` ou `screen` em treinos longos:

```bash
bash scripts/run_experiment.sh 1   # roda o pipeline completo e sai, sem esperar ENTER
```

Cada execução gera automaticamente:

- Um log completo por etapa em `logs/`
- Uma linha em `results/execution_times.csv` com timestamp, duração, status e commit do git — útil para reportar custo computacional na dissertação

No pipeline completo (opção 1), a falha de uma etapa **não interrompe** as demais: só são puladas as que dependem dela (N-HiTS de resíduos e Híbrido ARIMA dependem do ARIMA; gráficos dependem da avaliação). Ao final é gravado `results/run_summary_<data>.txt` com as etapas que falharam ou foram puladas e as linhas de erro (❌), aviso (⚠) e alerta de viés (🚨) dos logs. O código de saída é diferente de 0 se algo falhou.

## Testes

```bash
poetry run pytest -q
```

Testes com dados sintéticos para métricas, normalização, baselines ingênuos, testes estatísticos (Diebold-Mariano, Friedman, Holm, Nemenyi) e alinhamento/ausência de vazamento do protocolo `rolling` (inclui treinos reduzidos de statsforecast, Prophet e NeuralForecast; cerca de 3 minutos).

---

# 📂 Estrutura do Projeto

```text
pipeline-forecast/
│
├── config/
│   └── main_config.yaml
│
├── data/
│   ├── raw/
│   │   └── Datasets exatamente como baixados da fonte
│   └── bronze/
│       └── Datasets filtrados (date_column + target_column)
│
├── logs/
│   └── Log completo de cada etapa executada pelo orquestrador
│
├── results/
│   ├── forecasts/
│   │   ├── arima/
│   │   ├── deep/
│   │   ├── ets/
│   │   ├── hybrid/
│   │   ├── hybrid_blocks/
│   │   ├── hybrid_prophet/
│   │   └── prophet/
│   │
│   ├── metrics/
│   ├── plots/
│   ├── residuals/
│   └── execution_times.csv
│
├── src/
│   ├── ingestion/
│   ├── metrics/
│   ├── models/
│   └── visualization/
│
├── .env
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── scripts/
    ├── run_experiment.sh
    └── setup_venv.sh
```

---

# 📁 Diretórios

| Diretório | Descrição |
| ------------ | ----------- |
| `config/` | Configurações centrais do experimento |
| `data/raw/` | Datasets exatamente como baixados da fonte, sem tratamento |
| `data/bronze/` | Datasets filtrados para `date_column` + `target_column`, prontos para uso |
| `logs/` | Log completo (stdout/stderr) de cada etapa executada pelo orquestrador |
| `results/` | Resultados gerados pelo pipeline |
| `results/forecasts/` | Previsões produzidas por cada modelo |
| `results/metrics/` | Métricas consolidadas |
| `results/plots/` | Gráficos produzidos |
| `results/residuals/` | Resíduos utilizados pelos modelos híbridos |
| `results/execution_times.csv` | Tempo, status e commit do git de cada etapa executada |
| `src/` | Código-fonte do projeto |

---

# 📊 Fluxo do Pipeline

```text
Datasets
     │
     ▼
Ingestão (raw → bronze)
     │
     ▼
Modelagem Estatística
     │
     ├── AutoARIMA
     ├── AutoETS
     └── Prophet
     │
     ▼
Extração de Resíduos
     │
     ▼
Modelos Deep Learning
     │
     ├── N-HiTS
     ├── LSTM
     ├── Informer
     └── MLP
     │
     ▼
Modelos Híbridos
     │
     ▼
Avaliação
     │
     ▼
Métricas + Gráficos
```

---

# 📌 Objetivo da Pesquisa

Este projeto implementa um pipeline MLOps reprodutível para avaliar modelos estatísticos, modelos de Deep Learning e arquiteturas híbridas para previsão univariada de séries temporais (variável-alvo `OT`, temperatura do óleo, nos datasets ETT).

A principal contribuição consiste na utilização de modelagem estatística para capturar a componente linear da série e modelos neurais para modelar os resíduos, permitindo comparar diferentes estratégias híbridas contra o melhor modelo individual de cada dataset.

---

**Autor:** Flávio Monteiro
*Mestrando em Engenharia de Computação (PPGEC - UPE)*
