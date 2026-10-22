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

# 5. Verifica se o backend do Prophet (CmdStan) está funcional.
echo "----------------------------------------------------------------------"
echo "🔬 Verificando se o backend do Prophet (CmdStan) está funcional..."
set +e
poetry run python -c "from prophet import Prophet; Prophet()" > /dev/null 2>&1
PROPHET_OK=$?
set -e

if [ "$PROPHET_OK" -ne 0 ]; then
    echo "⚠️ Backend do Prophet não está funcional. Instalando toolchain C++ e CmdStan..."
    echo "   (isso pode levar alguns minutos e precisa de internet)"

    set +e
    poetry run python - <<'PYEOF'
import glob
import os
import sys
import cmdstanpy

print("Instalando toolchain C++ (MinGW/RTools)...")
try:
    cmdstanpy.install_cxx_toolchain()
except AttributeError:
    print("Aviso: install_cxx_toolchain obsoleto nesta versão. O install_cmdstan cuidará disso.")

home = os.path.expanduser("~")
candidatos = (
    glob.glob(os.path.join(home, ".cmdstan", "RTools*", "mingw64", "bin"))
    + glob.glob(os.path.join(home, ".cmdstan", "RTools*", "usr", "bin"))
    + glob.glob(os.path.join(home, ".cmdstan", "RTools*", "*", "mingw64", "bin"))
)
for caminho in candidatos:
    if caminho not in os.environ.get("PATH", ""):
        os.environ["PATH"] = caminho + os.pathsep + os.environ.get("PATH", "")
        print(f"PATH atualizado nesta sessão com: {caminho}")

if not candidatos:
    print("⚠️ Não encontrei automaticamente a pasta do toolchain em ~/.cmdstan/RTools*.")

print("Instalando CmdStan...")
cmdstanpy.install_cmdstan(overwrite=True)
print("OK: toolchain e CmdStan instalados nesta sessão.")
PYEOF
    TOOLCHAIN_STATUS=$?

    if [ "$TOOLCHAIN_STATUS" -ne 0 ]; then
        echo "❌ A instalação do toolchain/CmdStan falhou (veja o erro acima)."
        echo "   Se o erro ainda for 'mingw32-make não encontrado', feche e reabra"
        echo "   o terminal (para o PATH gravado pelo Windows via setx ser lido"
        echo "   numa sessão nova) e rode este script de novo."
    fi

    poetry run python -c "from prophet import Prophet; Prophet()" > /dev/null 2>&1
    PROPHET_OK=$?
    set -e

    if [ "$PROPHET_OK" -eq 0 ]; then
        echo "✅ Backend do Prophet corrigido com sucesso!"
    else
        echo "❌ Não foi possível corrigir o backend do Prophet automaticamente."
        echo "   No Windows, feche e reabra o terminal e rode este script de novo."
    fi
else
    echo "✅ Backend do Prophet OK."
fi

# 6. Estrutura de Diretórios e Configurações Padrões
echo "----------------------------------------------------------------------"
echo "📂 Criando estrutura de diretórios e arquivos de configuração padrão..."

mkdir -p logs results

if [ ! -f "results/execution_times.csv" ]; then
    echo "timestamp,step,duration_seconds,status,git_commit" > "results/execution_times.csv"
    echo "   -> Arquivo results/execution_times.csv inicializado."
fi

if [ ! -f ".env" ]; then
    cat <<EOT > .env
# Configurações Globais do Pipeline
TF_ENABLE_ONEDNN_OPTS=0
PYTHONIOENCODING="utf-8"
PYTHONUTF8=1
EOT
    echo "   -> Arquivo .env com variáveis padrões criado."
fi

echo "----------------------------------------------------------------------"
echo "✅ Ambiente virtual criado e dependências instaladas com sucesso!"
echo ""
echo "🔎 Interpretador usado pelo Poetry (bom registrar na metodologia do artigo científico):"
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