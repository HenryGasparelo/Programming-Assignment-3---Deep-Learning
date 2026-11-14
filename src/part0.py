import os
from typing import Dict
import pandas as pd
import torch.nn as nn

from src.utils import seed_everything, get_device, count_parameters, save_json
from src.data import get_supervised_dataloaders
from src.models import build_model
from src.training import train_supervised
from src.metrics import (
    evaluate_model,
    plot_training_curves,
    plot_comparison_curves,
    plot_confusion_matrix,
    detect_overfitting_epoch,
    build_per_class_f1_table,
)


def run_part0(
    data_dir: str = "data",
    results_dir: str = "results",
    figures_dir: str = "docs/figures",
    checkpoints_dir: str = "checkpoints",
    seed: int = 42,
    epochs_cnn: int = 45,
    epochs_resnet: int = 40,
    batch_size: int = 16,
    show_plots: bool = False,
) -> Dict:
    """
    Executa na íntegra a Parte 0 (O Piso):
      - Versão (a): SmallCNN treinada do zero com rotulos_5.csv (105 imagens)
      - Versão (b): ResNet-18 pré-treinada na ImageNet com fine-tuning em rotulos_5.csv (105 imagens)
    Salva checkpoints, curvas de treino/validação, matriz de confusão e tabelas de métricas.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = get_device(prefer_gpu=True)
    print(f"=== Executando Parte 0 (O Piso) | Dispositivo: {device} | Seed: {seed} ===\n")

    # 1. Versão (a): Pequena CNN treinada do zero (SmallCNN)
    print("--- [Parte 0 (a)] Treinando SmallCNN do zero (105 imagens) ---")
    seed_everything(seed)
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class, class_names = get_supervised_dataloaders(
        data_dir=data_dir, budget=5, batch_size=batch_size, img_size=224
    )

    model_a = build_model(arch="small_cnn", num_classes=len(class_names), pretrained=False, dropout=0.3, verbose=True)
    ckpt_a = os.path.join(checkpoints_dir, f"part0a_small_cnn_b5_s{seed}.pt")

    model_a, history_a, best_ep_a, best_val_f1_a = train_supervised(
        model=model_a,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs_cnn,
        lr=1e-3,
        backbone_lr_mult=1.0,
        weight_decay=1e-4,
        label_smoothing=0.05,
        num_classes=len(class_names),
        arch_name="small_cnn",
        checkpoint_path=ckpt_a,
        verbose=True,
    )

    eval_criterion = nn.CrossEntropyLoss()
    test_metrics_a = evaluate_model(model_a, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))
    diag_a = detect_overfitting_epoch(history_a)

    # 2. Versão (b): ResNet-18 pré-treinada na ImageNet com Fine-Tuning
    print("\n--- [Parte 0 (b)] Treinando ResNet-18 (ImageNet Fine-Tuning, 105 imagens) ---")
    seed_everything(seed)
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class, class_names = get_supervised_dataloaders(
        data_dir=data_dir, budget=5, batch_size=batch_size, img_size=224
    )

    model_b = build_model(arch="resnet18", num_classes=len(class_names), pretrained=True, dropout=0.2, verbose=True)
    ckpt_b = os.path.join(checkpoints_dir, f"part0b_resnet18_b5_s{seed}.pt")

    model_b, history_b, best_ep_b, best_val_f1_b = train_supervised(
        model=model_b,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs_resnet,
        lr=5e-4,
        backbone_lr_mult=0.2,
        weight_decay=1e-4,
        label_smoothing=0.05,
        num_classes=len(class_names),
        arch_name="resnet18",
        checkpoint_path=ckpt_b,
        verbose=True,
    )

    test_metrics_b = evaluate_model(model_b, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))
    diag_b = plot_training_curves(
        history=history_b,
        title_prefix="Parte 0 (b) — ResNet-18 (ImageNet Fine-Tuning, 105 rótulos)",
        save_path=os.path.join(figures_dir, "part0_curves_resnet18.png"),
        show=show_plots,
    )

    plot_comparison_curves(
        history_a=history_a,
        history_b=history_b,
        save_path=os.path.join(figures_dir, "part0_comparison_a_vs_b.png"),
        show=show_plots,
    )

    plot_confusion_matrix(
        cm=test_metrics_b["confusion_matrix"],
        class_names=class_names,
        title=(
            f"Matriz de Confusão — Parte 0 (b) ResNet-18 no test_public\n"
            f"(F1 Macro = {test_metrics_b['f1_macro']:.4f} | Acurácia = {test_metrics_b['accuracy']:.4f})"
        ),
        normalize=True,
        save_path=os.path.join(figures_dir, "part0_confusion_matrix_resnet18.png"),
        show=show_plots,
    )

    df_test = pd.read_csv(os.path.join(data_dir, "labels", "test_public.csv"))
    df_per_class = build_per_class_f1_table(
        class_names=class_names,
        df_test=df_test,
        test_metrics_a=test_metrics_a,
        test_metrics_b=test_metrics_b,
        save_path=os.path.join(results_dir, "part0_per_class_f1.csv"),
    )

    summary = {
        "seed": seed,
        "budget_per_class": 5,
        "total_labeled_train": 105,
        "ceiling_f1_macro_reference": 0.686,
        "part0_a_small_cnn": {
            "arch": "small_cnn",
            "pretrained": False,
            "num_parameters": count_parameters(model_a),
            "best_val_epoch": best_ep_a,
            "val_f1_macro": best_val_f1_a,
            "test_public_f1_macro": test_metrics_a["f1_macro"],
            "test_public_accuracy": test_metrics_a["accuracy"],
            "overfitting_diagnostics": diag_a,
            "history": history_a,
        },
        "part0_b_resnet18": {
            "arch": "resnet18",
            "pretrained": True,
            "num_parameters": count_parameters(model_b),
            "best_val_epoch": best_ep_b,
            "val_f1_macro": best_val_f1_b,
            "test_public_f1_macro": test_metrics_b["f1_macro"],
            "test_public_accuracy": test_metrics_b["accuracy"],
            "overfitting_diagnostics": diag_b,
            "history": history_b,
        },
    }
    save_json(summary, os.path.join(results_dir, "part0_summary.json"))

    return {
        "summary": summary,
        "df_per_class": df_per_class,
        "test_metrics_a": test_metrics_a,
        "test_metrics_b": test_metrics_b,
    }


if __name__ == "__main__":
    run_part0(show_plots=False)
