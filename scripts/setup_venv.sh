#!/bin/bash

# Encerra o script imediatamente se algum comando falhar
set -e

clear
echo "======================================================="
echo "   🛠️  Configuração do Ambiente MLOps (Poetry)"
echo "======================================================="

# 1. Verifica se o Python está instalado (Compatível com Linux e Windows)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ ERRO: Python não encontrado! Por favor, instale o Python no seu sistema (Ubuntu/Windows 11)."
    exit 1
fi

# 2. Verifica se o Poetry está instalado
if ! command -v poetry &> /dev/null; then
    echo "⚠️ Poetry não encontrado. Tentando instalar via pipx..."
    $PYTHON_CMD -m pip install --user pipx
    $PYTHON_CMD -m pipx install poetry
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
echo "👉 Para ativar o ambiente manualmente no Linux (Ubuntu/Fedora/Mac):"
echo "    source .venv/bin/activate"
echo "👉 Para ativar o ambiente manualmente no Windows (Git Bash):"
echo "    source .venv/Scripts/activate"
echo ""
echo "👉 Para iniciar o pipeline, rode o orquestrador:"
echo "    bash run_experiment.sh"
echo "======================================================="