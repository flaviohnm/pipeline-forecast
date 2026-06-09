#!/bin/bash

# Encerra o script imediatamente se algum comando falhar
set -e

clear
echo "======================================================="
echo "   🛠️  Configuração do Ambiente MLOps (Poetry)"
echo "======================================================="

# 1. Verifica se o Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ ERRO: Python3 não encontrado! Por favor, instale o Python no seu Fedora."
    exit 1
fi

# 2. Verifica se o Poetry está instalado
if ! command -v poetry &> /dev/null; then
    echo "⚠️ Poetry não encontrado. Tentando instalar via pip..."
    pip install --user pipx
    pipx install poetry
    echo "✅ Poetry instalado! (Pode ser necessário reiniciar o terminal ou rodar 'source ~/.bashrc')"
fi

# 3. Força o Poetry a criar o ambiente virtual DENTRO da pasta do projeto (.venv)
echo "⚙️ Configurando o Poetry para criar a pasta .venv localmente..."
poetry config virtualenvs.in-project true

# 4. Cria o ambiente e instala as dependências
echo "📦 Instalando/Sincronizando dependências do pyproject.toml..."
poetry install --no-root

echo "----------------------------------------------------------------------"
echo "✅ Ambiente virtual criado e dependências instaladas com sucesso!"
echo ""
echo "👉 Para ativar o ambiente manualmente, use:"
echo "    source .venv/bin/activate"
echo ""
echo "👉 Para iniciar o pipeline, rode o orquestrador:"
echo "    ./run_experiment.sh"
echo "======================================================="