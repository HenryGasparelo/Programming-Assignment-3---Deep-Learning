import os
import json
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    """Fixa todas as sementes aleatórias para garantir reprodutibilidade."""
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
    """Conta o número total de parâmetros do modelo conforme regra do enunciado."""
    return sum(p.numel() for p in model.parameters())


def check_parameter_limit(model: nn.Module, max_params: int = 12_000_000, verbose: bool = True) -> bool:
    """Verifica se o modelo respeita o teto de 12 milhões de parâmetros."""
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
    extra_info: dict = None,
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


def save_json(data: dict, path: str) -> None:
    """Salva um dicionário em arquivo JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
