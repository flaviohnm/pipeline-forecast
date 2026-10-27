"""
Escolha do dispositivo de treino das redes neurais (GPU CUDA ou CPU).

Regra: `compute.accelerator` em config/main_config.yaml
  - "auto" (padrão): usa a GPU se o PyTorch enxergar uma GPU CUDA; caso contrário, CPU.
  - "cpu": força CPU (útil para reproduzir resultados numa máquina sem GPU).
  - "gpu": exige GPU; se não houver, cai para CPU com aviso (o pipeline não é interrompido).
A mesma base de código roda, portanto, no Windows/CPU e no Ubuntu/GPU sem alterações.

Os modelos do NeuralForecast recebem `accelerator` e `devices` (kwargs do Trainer do Lightning).
"""

import os
import platform
from functools import lru_cache

ACCELERATOR_OPTIONS = ("auto", "cpu", "gpu")


@lru_cache(maxsize=1)
def cuda_info():
    """Informações da GPU CUDA visíveis ao PyTorch: dict(available, name, count, torch, cuda, reason)."""
    info = {"available": False, "name": None, "count": 0, "torch": None, "cuda": None, "reason": ""}
    try:
        import torch
    except Exception as exc:  # PyTorch ausente ou quebrado
        info["reason"] = f"PyTorch indisponível ({exc.__class__.__name__})"
        return info

    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        info["reason"] = "CUDA_VISIBLE_DEVICES vazio (GPU ocultada de propósito)"
        return info
    if info["cuda"] is None:
        info["reason"] = "PyTorch instalado sem suporte a CUDA (build somente CPU)"
        return info
    try:
        available = torch.cuda.is_available()
    except Exception as exc:
        info["reason"] = f"falha ao consultar a GPU ({exc.__class__.__name__})"
        return info
    if not available:
        info["reason"] = "nenhuma GPU CUDA detectada (ou driver NVIDIA incompatível com este PyTorch)"
        return info

    info.update(available=True, count=torch.cuda.device_count(), name=torch.cuda.get_device_name(0), reason="")
    return info


def resolve_accelerator(preference="auto", cuda=None):
    """
    Retorna ("gpu" | "cpu", motivo). `cuda` permite injetar o resultado de cuda_info() (testes).
    """
    if preference not in ACCELERATOR_OPTIONS:
        raise ValueError(f"compute.accelerator inválido: '{preference}'. Use um de {ACCELERATOR_OPTIONS}.")
    if preference == "cpu":
        return "cpu", "forçado por compute.accelerator = cpu"

    cuda = cuda if cuda is not None else cuda_info()
    if cuda["available"]:
        return "gpu", f"GPU CUDA detectada: {cuda['name']}"
    if preference == "gpu":
        return "cpu", f"compute.accelerator = gpu, mas {cuda['reason']}; usando CPU"
    return "cpu", cuda["reason"] or "nenhuma GPU disponível"


def get_trainer_kwargs(config=None, cuda=None):
    """
    kwargs de dispositivo para os modelos do NeuralForecast: {"accelerator": "gpu"|"cpu", "devices": n}.
    `config` é o dict lido de main_config.yaml (usa a seção `compute`).
    """
    compute = (config or {}).get("compute", {}) or {}
    accelerator, _ = resolve_accelerator(compute.get("accelerator", "auto"), cuda)
    devices = int(compute.get("devices", 1)) if accelerator == "gpu" else 1
    return {"accelerator": accelerator, "devices": devices}


def describe_compute(config=None, cuda=None):
    """Linha de log com o dispositivo usado, para registrar na metodologia e nos logs de cada execução."""
    compute = (config or {}).get("compute", {}) or {}
    cuda = cuda if cuda is not None else cuda_info()
    accelerator, reason = resolve_accelerator(compute.get("accelerator", "auto"), cuda)
    kwargs = get_trainer_kwargs(config, cuda)
    torch_txt = f"torch {cuda['torch']}" + (f" | CUDA {cuda['cuda']}" if cuda["cuda"] else "")
    prefix = "⚠️ " if (accelerator == "cpu" and compute.get("accelerator", "auto") == "gpu") else "🖥️ "
    return (
        f"{prefix}Dispositivo de treino: {accelerator.upper()} (devices={kwargs['devices']}) - {reason} "
        f"| {torch_txt} | {platform.system()} {platform.machine()}"
    )


if __name__ == "__main__":
    # Diagnóstico: poetry run python src/utils/compute.py
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent.parent
    with open(root / "config" / "main_config.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    print(describe_compute(cfg))
    gpu = cuda_info()
    if gpu["available"]:
        print(f"   GPUs visíveis: {gpu['count']} | GPU 0: {gpu['name']}")
    else:
        print(f"   Motivo de não haver GPU: {gpu['reason'] or 'compute.accelerator = cpu'}")
        print("   Se a máquina tem GPU NVIDIA: confira `nvidia-smi` (driver) e `poetry run python -c \"import torch;print(torch.version.cuda)\"`.")
        print("   O poetry.lock fixa um PyTorch com CUDA 13 no Linux, que exige driver NVIDIA recente (580 ou superior).")
        print("   Com driver mais antigo, o pipeline roda normalmente, mas na CPU.")
