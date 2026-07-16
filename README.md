# 🚀 Pipeline MLOps para Time Series Forecasting (Híbrido)

Este repositório contém o código-fonte e o pipeline MLOps desenvolvido para avaliar modelos estatísticos, de **Deep Learning** e arquiteturas híbridas na previsão de séries temporais multivariadas (*Time Series Forecasting*).

O projeto automatiza todo o ciclo de vida do experimento, desde a ingestão de dados de benchmarks renomados (ETT, Weather, ECL e AirPassengers), passando pelo treinamento dos modelos, extração de resíduos, inferência e avaliação multi-horizonte.

O objetivo da pesquisa é comparar o desempenho dos modelos híbridos desenvolvidos com os melhores modelos individuais em cada conjunto de dados.

---

# 🎯 Arquitetura e Modelos

A proposta metodológica utiliza modelos estatísticos para capturar os componentes lineares da série temporal e redes neurais profundas para modelar os resíduos não lineares.

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

- Python
  - **3.11.9** ou superior

- Gerenciador de Dependências
  - Poetry

---

# 🛠️ Instalação

## 1. Conceda permissão aos scripts

```bash
chmod +x setup_venv.sh run_experiment.sh
```

## 2. Crie o ambiente virtual

```bash
./setup_venv.sh
```

O script realiza automaticamente:

- Verificação da versão do Python;
- Instalação do Poetry (caso necessário);
- Criação do ambiente `.venv`;
- Instalação de todas as dependências do projeto.

---

# ⚙️ Configuração

Toda a configuração do experimento é feita através do arquivo `.env`.

Exemplo:

```env
# Opções:
# ETTh1
# ETTh2
# ETTm1
# ECL
# Weather
# AirPassengers
# ALL

DATASET=Weather
```

---

# 🚀 Executando o Pipeline

Execute o orquestrador:

```bash
./run_experiment.sh
```

Será exibido o seguinte menu:

```text
=======================================================
🚀 Time Series Forecasting - Orquestrador Híbrido
=======================================================

✅ Ambiente gerenciado pelo Poetry.
🎯 DATASET ATIVO (Lido do .env): Weather

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

---

# 📂 Estrutura do Projeto

```text
pipeline-forecast/
│
├── config/
│   └── main_config.yaml
│
├── data/
│   └── Datasets baixados
│
├── results/
│   ├── forecasts/
│   │   ├── arima/
│   │   ├── deep/
│   │   ├── ets/
│   │   ├── hybrid/
│   │   ├── hybrid_prophet/
│   │   └── prophet/
│   │
│   ├── metrics/
│   ├── plots/
│   └── residuals/
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
├── run_experiment.sh
└── setup_venv.sh
```

---

# 📁 Diretórios

| Diretório | Descrição |
|------------|-----------|
| `config/` | Configurações centrais do experimento |
| `data/` | Datasets utilizados |
| `results/` | Resultados gerados pelo pipeline |
| `results/forecasts/` | Previsões produzidas por cada modelo |
| `results/metrics/` | Métricas consolidadas |
| `results/plots/` | Gráficos produzidos |
| `results/residuals/` | Resíduos utilizados pelos modelos híbridos |
| `src/` | Código-fonte do projeto |

---

# 📊 Fluxo do Pipeline

```text
Datasets
     │
     ▼
Ingestão
     │
     ▼
Modelos Estatísticos
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

Este projeto implementa um pipeline MLOps reprodutível para avaliar modelos estatísticos, modelos de Deep Learning e arquiteturas híbridas para previsão de séries temporais multivariadas.

A principal contribuição consiste na utilização de modelos estatísticos para capturar a componente linear da série e modelos neurais para modelar os resíduos, permitindo comparar diferentes estratégias híbridas em múltiplos datasets públicos.