#!/bin/bash

# Encerra o script imediatamente se algum comando falhar
set -e

clear
echo "======================================================="
echo "   🛠️  Configuração do Ambiente MLOps (Poetry)"
echo "======================================================="

# 0. Verifica se existe um pyproject.toml no diretório atual
if [ ! -f "pyproject.toml" ]; then
    echo "❌ ERRO: Nenhum pyproject.toml encontrado neste diretório."
    echo "   Rode este script a partir da raiz do projeto."
    exit 1
fi

# 1. Verifica se o Python está instalado (Compatível com Linux e Windows)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ ERRO: Python não encontrado! Por favor, instale o Python no seu sistema (Ubuntu/Windows 11)."
    exit 1
fi
echo "🐍 Usando: $($PYTHON_CMD --version)"

# 2. Verifica se o Poetry está instalado
if ! command -v poetry &> /dev/null; then
    echo "⚠️ Poetry não encontrado. Tentando instalar via pipx..."
    $PYTHON_CMD -m pip install --user pipx
    $PYTHON_CMD -m pipx install poetry
    echo "✅ Poetry instalado! (Pode ser necessário reiniciar o terminal ou rodar 'source ~/.bashrc')"

    if ! command -v poetry &> /dev/null; then
        echo "⚠️ 'poetry' ainda não está no PATH desta sessão."
        echo "   Rode 'source ~/.bashrc' (ou reabra o terminal) e execute este script novamente."
        exit 1
    fi
fi
echo "📦 Poetry: $(poetry --version)"

# 3. Força o Poetry a criar o ambiente virtual DENTRO da pasta do projeto (.venv)
echo "⚙️ Configurando o Poetry para criar a pasta .venv localmente..."
poetry config virtualenvs.in-project true

# 4. Cria o ambiente e instala as dependências
echo "📦 Instalando/Sincronizando dependências do pyproject.toml..."
poetry install --no-root

echo "----------------------------------------------------------------------"
echo "✅ Ambiente virtual criado e dependências instaladas com sucesso!"
echo ""
echo "🔎 Interpretador usado pelo Poetry (bom registrar na metodologia da dissertação):"
poetry env info --path 2>/dev/null || true
poetry run python --version
echo ""
echo "👉 Para ativar o ambiente manualmente no Linux (Ubuntu/Fedora/Mac):"
echo "    source .venv/bin/activate"
echo "👉 Para ativar o ambiente manualmente no Windows (Git Bash):"
echo "    source .venv/Scripts/activate"
echo ""
echo "👉 Para iniciar o pipeline, rode o orquestrador:"
echo "    bash run_experiment.sh"
echo "👉 Para rodar sem menu interativo (ex.: em background/servidor remoto):"
echo "    nohup bash run_experiment.sh 1 > full_run.log 2>&1 &"
echo "======================================================="