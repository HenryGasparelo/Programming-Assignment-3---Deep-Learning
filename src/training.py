import copy
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from src.metrics import evaluate_model
from src.utils import save_checkpoint


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 40,
    lr: float = 3e-4,
    backbone_lr_mult: float = 0.2,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    num_classes: int = 21,
    arch_name: str = "resnet18",
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]], int, float]:
    """
    Treina um classificador supervisionado, registrando Loss, Acurácia e F1 Macro por época
    tanto no conjunto de treino quanto no conjunto de validação (`val`).
    A seleção da melhor época é feita ESTRITAMENTE pelo maior F1 Macro no `val`.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    # Taxa de aprendizado diferenciada se for um backbone pré-treinado (camada fc aprende mais rápido)
    if hasattr(model, "fc") and backbone_lr_mult != 1.0:
        fc_params = list(model.fc.parameters())
        fc_param_ids = {id(p) for p in fc_params}
        backbone_params = [p for p in model.parameters() if id(p) not in fc_param_ids and p.requires_grad]
        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": lr * backbone_lr_mult},
                {"params": fc_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_f1 = -1.0
    best_epoch = 1
    best_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        train_preds = []
        train_targets = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            preds = torch.argmax(logits.detach(), dim=1)
            train_preds.append(preds.cpu().numpy())
            train_targets.append(targets.cpu().numpy())

        scheduler.step()

        y_train_true = np.concatenate(train_targets)
        y_train_pred = np.concatenate(train_preds)
        train_loss = running_loss / max(total_samples, 1)
        train_f1 = float(
            f1_score(y_train_true, y_train_pred, average="macro", labels=list(range(num_classes)), zero_division=0)
        )
        train_acc = float(accuracy_score(y_train_true, y_train_pred))

        # Avaliação no conjunto de validação (val)
        val_metrics = evaluate_model(model, val_loader, device, criterion=eval_criterion, num_classes=num_classes)
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1_macro"]
        val_acc = val_metrics["accuracy"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                save_checkpoint(
                    model=model,
                    path=checkpoint_path,
                    arch=arch_name,
                    num_classes=num_classes,
                    epoch=epoch,
                    val_f1=val_f1,
                )

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs or is_best):
            marker = " ★ [Melhor Val F1]" if is_best else ""
            print(
                f"Época {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} (Acc: {train_acc:.3f}) | "
                f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} (Acc: {val_acc:.3f}){marker}"
            )

    elapsed = time.time() - start_time
    if verbose:
        print(
            f"\nTreinamento concluído em {elapsed:.1f}s. "
            f"Melhor época no val: {best_epoch} (Val F1 Macro = {best_val_f1:.4f})."
        )

    model.load_state_dict(best_weights)
    return model, history, best_epoch, best_val_f1
