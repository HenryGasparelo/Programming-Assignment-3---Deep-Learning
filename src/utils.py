import os
import random
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    """Fixa todas as sementes aleatórias para garantir reprodutibilidade estrita."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Retorna o dispositivo de execução (CUDA, MPS ou CPU)."""
    if not prefer_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Conta o número total de parâmetros do modelo conforme regra oficial do enunciado."""
    return sum(p.numel() for p in model.parameters())


def check_parameter_limit(model: nn.Module, max_params: int = 12_000_000, verbose: bool = True) -> bool:
    """Verifica se o modelo respeita o teto estrito de 12 milhões de parâmetros."""
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total de parâmetros: {total_params:,} (Treináveis: {trainable_params:,})")
        if total_params <= max_params:
            print(f"✓ Aprovado: O modelo cumpre a restrição de <= {max_params:,} parâmetros.")
        else:
            print(f"✗ Reprovado: O modelo excede o limite permitido ({total_params:,} > {max_params:,})!")
    return total_params <= max_params


def save_checkpoint(
    model: nn.Module,
    path: str,
    arch: str = "resnet18",
    num_classes: int = 21,
    epoch: int = 0,
    val_f1: float = 0.0,
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Salva os pesos do modelo e metadados no caminho especificado."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "arch": arch,
        "num_classes": num_classes,
        "epoch": epoch,
        "val_f1": val_f1,
        "num_parameters": count_parameters(model),
    }
    if extra_info:
        payload.update(extra_info)
    torch.save(payload, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Carrega os pesos de um checkpoint salvo no modelo fornecido."""
    if device is None:
        device = get_device(prefer_gpu=True)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, ckpt

