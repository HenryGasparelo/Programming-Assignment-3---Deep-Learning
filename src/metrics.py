import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.utils import count_parameters


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    num_classes: int = 21,
) -> Dict:
    """
    Avalia o modelo em um DataLoader e retorna loss, F1 Macro, acurácia, F1 por classe,
    rótulos verdadeiros, predições e probabilidades.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_probs = []

    for batch in loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        logits = model(inputs)
        if criterion is not None:
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_samples += inputs.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    avg_loss = total_loss / max(total_samples, 1) if criterion is not None else 0.0
    f1_macro = float(f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "loss": avg_loss,
        "f1_macro": f1_macro,
        "accuracy": acc,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def detect_overfitting_epoch(history: Dict[str, List[float]], train_acc_threshold: float = 0.90) -> Dict[str, int]:
    """
    Analisa as curvas de treino e validação para identificar:
    1) A época em que a loss de validação atinge o mínimo.
    2) A época em que a acurácia/F1 de treino ultrapassa `train_acc_threshold` (ex: 90%), caracterizando memorização.
    3) A época de melhor F1 macro no conjunto de validação (`val`).
    """
    epochs = history["epoch"]
    val_losses = history["val_loss"]
    val_f1s = history["val_f1"]
    train_accs = history["train_acc"]
    train_f1s = history["train_f1"]

    min_val_loss_idx = int(np.argmin(val_losses))
    best_val_f1_idx = int(np.argmax(val_f1s))

    memorization_epoch = epochs[-1]
    for ep, t_acc, t_f1 in zip(epochs, train_accs, train_f1s):
        if t_acc >= train_acc_threshold or t_f1 >= train_acc_threshold:
            memorization_epoch = ep
            break

    return {
        "min_val_loss_epoch": epochs[min_val_loss_idx],
        "best_val_f1_epoch": epochs[best_val_f1_idx],
        "memorization_90pct_epoch": memorization_epoch,
    }


def plot_training_curves(
    history: Dict[str, List[float]],
    title_prefix: str = "Parte 0 (b) — ResNet-18 (ImageNet Fine-Tuning)",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Dict[str, int]:
    """
    Plota as curvas de Loss e F1 Macro (Treino vs. Validação), destacando o ponto de início de memorização
    (overfitting) e a época selecionada pelo melhor F1 Macro na validação.
    """
    diag = detect_overfitting_epoch(history)
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=130)

    # Subplot 1: Loss (Treino vs Val)
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Treino (105 imgs)", color="#1f77b4", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validação (202 imgs)", color="#d62728", linewidth=2)
    ax.axvline(
        diag["min_val_loss_epoch"],
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        label=f"Mín. Val Loss (Época {diag['min_val_loss_epoch']})",
    )
    ax.axvline(
        diag["memorization_90pct_epoch"],
        color="#9467bd",
        linestyle=":",
        linewidth=1.5,
        label=f"Treino >= 90% / Decorando (Época {diag['memorization_90pct_epoch']})",
    )
    ax.set_title(f"{title_prefix}\nCurva de Loss (Cross-Entropy)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Subplot 2: F1 Macro (Treino vs Val)
    ax = axes[1]
    ax.plot(epochs, history["train_f1"], label="F1 Macro Treino", color="#1f77b4", linewidth=2)
    ax.plot(epochs, history["val_f1"], label="F1 Macro Validação", color="#2ca02c", linewidth=2)
    best_ep = diag["best_val_f1_epoch"]
    best_val_f1 = history["val_f1"][best_ep - 1]
    ax.axvline(
        best_ep,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label=f"Melhor F1 Val: {best_val_f1:.3f} (Época {best_ep})",
    )
    ax.axvline(
        diag["memorization_90pct_epoch"],
        color="#9467bd",
        linestyle=":",
        linewidth=1.5,
        label=f"Início de Memorização (Época {diag['memorization_90pct_epoch']})",
    )
    ax.set_title(f"{title_prefix}\nCurva de F1 Macro (Treino vs. Validação)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("F1 Macro")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return diag


def plot_comparison_curves(
    history_a: Dict[str, List[float]],
    history_b: Dict[str, List[float]],
    label_a: str = "0(a) SmallCNN",
    label_b: str = "0(b) ResNet-18",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plota lado a lado a comparação de F1 Macro e Cross-Entropy Loss entre dois modelos/versões
    (ex: Parte 0a treinada do zero vs. Parte 0b com fine-tuning da ImageNet).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=130)

    axes[0].plot(history_a["epoch"], history_a["train_f1"], "--", color="#1f77b4", label=f"{label_a} - Treino")
    axes[0].plot(history_a["epoch"], history_a["val_f1"], "-", color="#1f77b4", linewidth=2, label=f"{label_a} - Val")
    axes[0].plot(history_b["epoch"], history_b["train_f1"], "--", color="#2ca02c", label=f"{label_b} - Treino")
    axes[0].plot(history_b["epoch"], history_b["val_f1"], "-", color="#2ca02c", linewidth=2, label=f"{label_b} - Val")
    axes[0].set_title("Comparação F1 Macro: Do Zero vs. ImageNet Fine-Tuning", fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("F1 Macro")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(history_a["epoch"], history_a["train_loss"], "--", color="#1f77b4", label=f"{label_a} - Treino")
    axes[1].plot(history_a["epoch"], history_a["val_loss"], "-", color="#1f77b4", linewidth=2, label=f"{label_a} - Val")
    axes[1].plot(history_b["epoch"], history_b["train_loss"], "--", color="#d62728", label=f"{label_b} - Treino")
    axes[1].plot(history_b["epoch"], history_b["val_loss"], "-", color="#d62728", linewidth=2, label=f"{label_b} - Val")
    axes[1].set_title("Comparação de Loss: Do Zero vs. ImageNet Fine-Tuning", fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Matriz de Confusão — Parte 0 (b) no test_public",
    normalize: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plota a matriz de confusão 21x21 com contagens absolutas e/ou proporções normalizadas por linha (classe real).
    """
    fig, ax = plt.subplots(figsize=(13, 11), dpi=130)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Recall da Classe (Proporção por Linha)"},
            ax=ax,
            annot_kws={"size": 8},
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            annot_kws={"size": 8},
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Classe Predita", fontsize=11, fontweight="bold")
    ax.set_ylabel("Classe Verdadeira ( Contagem Anotada | Cor = Recall )", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_part0_summary_table(
    model_a: nn.Module,
    best_ep_a: int,
    best_val_f1_a: float,
    test_metrics_a: Dict,
    model_b: nn.Module,
    best_ep_b: int,
    best_val_f1_b: float,
    test_metrics_b: Dict,
    ceiling_f1: float = 0.6860,
) -> pd.DataFrame:
    """Constrói o DataFrame comparativo entre Parte 0 (a), Parte 0 (b) e o Teto de Referência."""
    return pd.DataFrame([
        {
            "Modelo / Versão": "Parte 0 (a) — SmallCNN (Do Zero)",
            "Parâmetros": f"{count_parameters(model_a):,}",
            "Melhor Época (val)": best_ep_a,
            "Val F1 Macro": round(best_val_f1_a, 4),
            "Test Public F1 Macro": round(test_metrics_a["f1_macro"], 4),
            "Test Public Acurácia": round(test_metrics_a["accuracy"], 4),
        },
        {
            "Modelo / Versão": "Parte 0 (b) — ResNet-18 (ImageNet Fine-Tuning)",
            "Parâmetros": f"{count_parameters(model_b):,}",
            "Melhor Época (val)": best_ep_b,
            "Val F1 Macro": round(best_val_f1_b, 4),
            "Test Public F1 Macro": round(test_metrics_b["f1_macro"], 4),
            "Test Public Acurácia": round(test_metrics_b["accuracy"], 4),
        },
        {
            "Modelo / Versão": "Teto de Referência — ResNet-18 (6.505 rótulos)",
            "Parâmetros": f"{count_parameters(model_b):,}",
            "Melhor Época (val)": "-",
            "Val F1 Macro": "-",
            "Test Public F1 Macro": ceiling_f1,
            "Test Public Acurácia": "-",
        },
    ])


def build_per_class_f1_table(
    class_names: List[str],
    df_test: pd.DataFrame,
    test_metrics_a: Dict,
    test_metrics_b: Dict,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Constrói e opcionalmente salva o DataFrame de F1 por classe comparando Parte 0 (a) e Parte 0 (b)."""
    class_counts = df_test["label"].value_counts().to_dict()
    rows = []
    for idx, cname in enumerate(class_names):
        rows.append({
            "idx": idx,
            "label": cname,
            "test_support": int(class_counts.get(cname, 0)),
            "f1_part0a_small_cnn": float(test_metrics_a["per_class_f1"][idx]),
            "f1_part0b_resnet18": float(test_metrics_b["per_class_f1"][idx]),
            "gain_b_minus_a": float(test_metrics_b["per_class_f1"][idx] - test_metrics_a["per_class_f1"][idx]),
        })
    df_per_class = pd.DataFrame(rows)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_per_class.to_csv(save_path, index=False)
    return df_per_class
