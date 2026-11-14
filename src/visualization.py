from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import load_class_mapping, get_transforms
from src.metrics import detect_overfitting_epoch


def plot_training_curves(
    history: Dict[str, List[float]],
    title_prefix: str = "ResNet-18",
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, int]]:
    """
    Plota na hora as curvas de Loss e F1 Macro (Treino vs. Validação), destacando o início
    de memorização (overfitting), o ponto de mínima loss de validação e o melhor F1 Macro do `val`.
    """
    diag = detect_overfitting_epoch(history)
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=130)

    # Subplot 1: Loss (Treino vs Val)
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Treino", color="#1f77b4", linewidth=2)
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
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, diag


def plot_comparison_curves(
    history_a: Dict[str, List[float]],
    history_b: Dict[str, List[float]],
    label_a: str = "0(a) SmallCNN",
    label_b: str = "0(b) ResNet-18",
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plota na hora a comparação de F1 Macro e Loss entre dois modelos/versões
    (ex: SmallCNN do zero vs. ResNet-18 ImageNet Fine-Tuning).
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
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Matriz de Confusão no test_public",
    normalize: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plota na hora a matriz de confusão 21x21 com contagens e proporções normalizadas por recall."""
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

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_teacher_noise_diagnostics(
    zs_eval: Dict,
    teacher_cache: Dict,
    imp_teacher: Dict,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plota na hora o diagnóstico de ruído e viés do Professor Zero-Shot vs. Professor Adaptado."""
    class_names = teacher_cache["class_names"]
    num_classes = len(class_names)

    zs_preds_train = teacher_cache["train"]["logits_trap_prompt"].argmax(dim=1).numpy()
    imp_preds_train = imp_teacher["probs_train"].argmax(axis=1)

    zs_counts = np.bincount(zs_preds_train, minlength=num_classes)
    imp_counts = np.bincount(imp_preds_train, minlength=num_classes)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=130)

    y_pos = np.arange(num_classes)
    height = 0.38
    ax = axes[0]
    ax.barh(y_pos - height / 2, zs_counts, height=height, color="#d62728", alpha=0.85, label="Professor Zero-Shot (Bruto)")
    ax.barh(y_pos + height / 2, imp_counts, height=height, color="#1f77b4", alpha=0.85, label="Professor Melhorado (105 rótulos)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Nº de Imagens Preditas no Pool train (6.505 imgs)")
    ax.set_title("Viés de Pseudo-Rótulos no train:\nZero-Shot Bruto vs. Professor Melhorado", fontweight="bold", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    df_conf = zs_eval["df_top_confusions"].head(10).iloc[::-1]
    labels_conf = [f"{r['classe_verdadeira']} → {r['classe_predita_professor']}" for _, r in df_conf.iterrows()]
    ax2.barh(labels_conf, df_conf["erros"], color="#ff7f0e", alpha=0.9)
    for idx_bar, (_, r) in enumerate(df_conf.iterrows()):
        ax2.text(
            r["erros"] + 0.3,
            idx_bar,
            f"{r['erros']}/{r['suporte_classe']} ({r['taxa_confusao']*100:.0f}%)",
            va="center",
            fontsize=8.5,
        )
    ax2.set_xlim(0, max(df_conf["erros"]) * 1.25)
    ax2.set_xlabel("Nº de Erros no test_public")
    ax2.set_title("Top 10 Erros Sistemáticos na Matriz de Ruído\ndo Professor Zero-Shot (BioCLIP)", fontweight="bold", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_student_vs_teacher_per_class(
    class_names: List[str],
    f1_piso: np.ndarray,
    f1_teacher_zs: np.ndarray,
    f1_teacher_imp: np.ndarray,
    f1_student: np.ndarray,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Plota na hora o comparativo de F1 por classe: Piso vs. Professor Zero-Shot vs. Aluno."""
    rows = []
    for idx, cname in enumerate(class_names):
        rows.append({
            "idx": idx,
            "label": cname,
            "f1_piso_0b": float(f1_piso[idx]),
            "f1_prof_zeroshot": float(f1_teacher_zs[idx]),
            "f1_prof_melhorado": float(f1_teacher_imp[idx]),
            "f1_aluno_resnet18": float(f1_student[idx]),
            "ganho_aluno_vs_prof_zs": float(f1_student[idx] - f1_teacher_zs[idx]),
            "ganho_aluno_vs_prof_imp": float(f1_student[idx] - f1_teacher_imp[idx]),
            "ganho_aluno_vs_piso": float(f1_student[idx] - f1_piso[idx]),
        })
    df_comp = pd.DataFrame(rows)
    df_sorted = df_comp.sort_values("ganho_aluno_vs_prof_zs", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=130)

    y_pos = np.arange(len(class_names))
    h = 0.27
    ax = axes[0]
    ax.barh(y_pos - h, df_comp["f1_prof_zeroshot"], height=h, label="Professor Zero-Shot (BioCLIP)", color="#d62728", alpha=0.8)
    ax.barh(y_pos, df_comp["f1_prof_melhorado"], height=h, label="Professor Melhorado (105 rótulos)", color="#ff7f0e", alpha=0.85)
    ax.barh(y_pos + h, df_comp["f1_aluno_resnet18"], height=h, label="Aluno ResNet-18 (Trilha C)", color="#2ca02c", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_comp["label"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("F1 Score no test_public")
    ax.set_title("Desempenho por Classe: Professor vs. Aluno (ResNet-18)", fontweight="bold", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")

    ax2 = axes[1]
    ax2.barh(
        y_pos - 0.18,
        df_sorted["ganho_aluno_vs_prof_zs"],
        height=0.35,
        color="#1f77b4",
        alpha=0.85,
        label="Aluno − Professor Zero-Shot",
    )
    ax2.barh(
        y_pos + 0.18,
        df_sorted["ganho_aluno_vs_prof_imp"],
        height=0.35,
        color="#2ca02c",
        alpha=0.85,
        label="Aluno − Professor Melhorado",
    )
    ax2.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_sorted["label"], fontsize=9)
    ax2.set_xlabel("Diferença de F1 (ΔF1 no test_public)")
    ax2.set_title("Em Quais Classes o Aluno Supera o Professor? (ΔF1 > 0)", fontweight="bold", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, df_comp


def plot_budget_curve(
    df_curva: pd.DataFrame,
    teacher_zs_f1: float = 0.3017,
    ceiling_f1: float = 0.6860,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plota na hora o gráfico oficial da Parte 2:
    F1 Macro no `test_public` vs. número de rótulos por classe (eixo x em escala log),
    com Piso Supervisionado, Aluno Trilha C, barras de erro (3 seeds no B=5), Professor e Teto.
    """
    budgets = [1, 5, 20]

    def agg_method(method_name: str):
        sub = df_curva[df_curva["metodo"] == method_name]
        means_f1, stds_f1, means_acc, stds_acc = [], [], [], []
        for b in budgets:
            b_sub = sub[sub["orcamento"] == b]
            means_f1.append(float(b_sub["f1_macro"].mean()))
            stds_f1.append(float(b_sub["f1_macro"].std(ddof=1)) if len(b_sub) > 1 else 0.0)
            means_acc.append(float(b_sub["acuracia"].mean()))
            stds_acc.append(float(b_sub["acuracia"].std(ddof=1)) if len(b_sub) > 1 else 0.0)
        return np.array(means_f1), np.array(stds_f1), np.array(means_acc), np.array(stds_acc)

    piso_f1_m, piso_f1_s, piso_acc_m, piso_acc_s = agg_method("piso_resnet18")
    aluno_f1_m, aluno_f1_s, aluno_acc_m, aluno_acc_s = agg_method("trilha_c_aluno_resnet18")

    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=130)

    ax.axhline(
        ceiling_f1,
        color="#2b2b2b",
        linestyle="--",
        linewidth=2,
        label=f"Teto Supervisionado (6.505 rótulos): F1 = {ceiling_f1:.3f}",
    )

    ax.axhline(
        teacher_zs_f1,
        color="#d62728",
        linestyle=":",
        linewidth=2,
        label=f"Professor BioCLIP Zero-Shot (0 rótulos): F1 = {teacher_zs_f1:.3f}",
    )

    ax.errorbar(
        budgets,
        piso_f1_m,
        yerr=piso_f1_s,
        fmt="-o",
        color="#1f77b4",
        linewidth=2.4,
        markersize=8,
        capsize=5,
        label="Piso Supervisionado — ResNet-18 (ImageNet Fine-Tuning)",
    )

    ax.errorbar(
        budgets,
        aluno_f1_m,
        yerr=aluno_f1_s,
        fmt="-s",
        color="#2ca02c",
        linewidth=2.6,
        markersize=8,
        capsize=5,
        label="Trilha C — Aluno ResNet-18 (Professor + Filtro + Soft KD)",
    )

    for i, b in enumerate(budgets):
        p_txt = f"{piso_f1_m[i]:.3f}±{piso_f1_s[i]:.3f}" if piso_f1_s[i] > 0 else f"{piso_f1_m[i]:.3f}"
        a_txt = f"{aluno_f1_m[i]:.3f}±{aluno_f1_s[i]:.3f}" if aluno_f1_s[i] > 0 else f"{aluno_f1_m[i]:.3f}"
        ax.annotate(p_txt, (b, piso_f1_m[i]), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color="#1f77b4", fontweight="bold")
        ax.annotate(a_txt, (b, aluno_f1_m[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9.5, color="#1b781b", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels(["1 rótulo/classe\n(21 total)", "5 rótulos/classe\n(105 total — 3 seeds)", "20 rótulos/classe\n(420 total)"], fontsize=10)
    ax.set_xlim(0.75, 26)
    ax.set_ylim(0.18, 0.74)
    ax.set_xlabel("Orçamento de Rótulos Humanos (Escala Logarítmica)", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Macro no test_public", fontsize=11, fontweight="bold")
    ax.set_title("Parte 2 — Curva de Orçamento: F1 Macro no test_public vs. Nº de Rótulos", fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9.5, loc="lower right")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    summary_rows = []
    for i, b in enumerate(budgets):
        summary_rows.append({
            "Orçamento (por classe)": b,
            "Total de Rótulos": b * 21,
            "Seeds": "42, 123, 999" if b == 5 else "42",
            "Piso F1 Macro (test_public)": f"{piso_f1_m[i]:.4f} ± {piso_f1_s[i]:.4f}" if b == 5 else f"{piso_f1_m[i]:.4f}",
            "Aluno Trilha C F1 Macro (test_public)": f"{aluno_f1_m[i]:.4f} ± {aluno_f1_s[i]:.4f}" if b == 5 else f"{aluno_f1_m[i]:.4f}",
            "Ganho ΔF1 Macro (Aluno − Piso)": f"{(aluno_f1_m[i] - piso_f1_m[i]):+.4f}",
            "Piso Acurácia": f"{piso_acc_m[i]:.4f} ± {piso_acc_s[i]:.4f}" if b == 5 else f"{piso_acc_m[i]:.4f}",
            "Aluno Trilha C Acurácia": f"{aluno_acc_m[i]:.4f} ± {aluno_acc_s[i]:.4f}" if b == 5 else f"{aluno_acc_m[i]:.4f}",
        })
    df_agg_table = pd.DataFrame(summary_rows)
    return fig, ax, df_agg_table


def plot_per_class_gain_budget5(
    class_names: List[str],
    piso_b5_per_class_list: List[np.ndarray],
    aluno_b5_per_class_list: List[np.ndarray],
    data_dir: str = "data",
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Plota na hora a quebra horizontal de ganho líquido por classe e comparação de F1 absoluto (média de 3 seeds)."""
    df_test = pd.read_csv(os.path.join(data_dir, "labels", "test_public.csv"))
    support_map = df_test["label"].value_counts().to_dict()

    piso_mat = np.stack(piso_b5_per_class_list, axis=0)
    aluno_mat = np.stack(aluno_b5_per_class_list, axis=0)

    piso_mean = piso_mat.mean(axis=0)
    piso_std = piso_mat.std(axis=0, ddof=1)
    aluno_mean = aluno_mat.mean(axis=0)
    aluno_std = aluno_mat.std(axis=0, ddof=1)
    gain_mean = aluno_mean - piso_mean

    rows = []
    for idx, cname in enumerate(class_names):
        rows.append({
            "idx": idx,
            "label": cname,
            "test_support": int(support_map.get(cname, 0)),
            "piso_f1_mean_3seeds": float(piso_mean[idx]),
            "piso_f1_std_3seeds": float(piso_std[idx]),
            "aluno_f1_mean_3seeds": float(aluno_mean[idx]),
            "aluno_f1_std_3seeds": float(aluno_std[idx]),
            "ganho_f1_mean": float(gain_mean[idx]),
        })

    df_gain = pd.DataFrame(rows).sort_values("ganho_f1_mean", ascending=False).reset_index(drop=True)
    df_plot = df_gain.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=130)

    ax = axes[0]
    colors = ["#2ca02c" if g > 0.01 else ("#7f7f7f" if g >= -0.01 else "#d62728") for g in df_plot["ganho_f1_mean"]]
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, df_plot["ganho_f1_mean"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    for i, (_, r) in enumerate(df_plot.iterrows()):
        val = r["ganho_f1_mean"]
        offset = 0.008 if val >= 0 else -0.008
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:+.3f}", va="center", ha=ha, fontsize=8.5, fontweight="bold")
    ax.set_xlim(min(df_plot["ganho_f1_mean"]) - 0.12, max(df_plot["ganho_f1_mean"]) + 0.12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["label"], fontsize=9)
    ax.set_xlabel("Ganho Médio ΔF1 (Aluno Trilha C − Piso Supervisionado, 3 seeds)")
    ax.set_title("Quebra do Ganho por Classe no Orçamento de 5 Rótulos\n(Média de 3 Seeds no test_public)", fontweight="bold", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)

    ax2 = axes[1]
    h = 0.36
    ax2.barh(y_pos - h / 2, df_plot["piso_f1_mean_3seeds"], xerr=df_plot["piso_f1_std_3seeds"], height=h, color="#1f77b4", alpha=0.85, label="Piso Supervisionado (Média ± DP)")
    ax2.barh(y_pos + h / 2, df_plot["aluno_f1_mean_3seeds"], xerr=df_plot["aluno_f1_std_3seeds"], height=h, color="#2ca02c", alpha=0.85, label="Aluno Trilha C (Média ± DP)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_plot["label"], fontsize=9)
    ax2.set_xlabel("F1 Score no test_public (Orçamento 5, 3 Seeds)")
    ax2.set_title("F1 por Classe no Orçamento 5: Piso vs. Aluno Trilha C", fontweight="bold", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, df_gain


def plot_gradcam_gallery(
    model: nn.Module,
    image_ids: List[str],
    data_dir: str = "data",
    split: str = "test_public",
    expected_correct: Optional[bool] = None,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """
    Calcula o Grad-CAM na última camada convolucional (`layer4`) da ResNet-18 para uma lista
    fixa de `image_ids` e plota na hora a imagem original ao lado do mapa de saliência (Grad-CAM).
    Mantém o gráfico limpo (apenas ID, classe real, classe predita e confiança) para que os
    diagnósticos detalhados fiquem no Markdown.
    """
    device = next(model.parameters()).device
    model.eval()

    class_to_idx, idx_to_class, _ = load_class_mapping(os.path.join(data_dir, "classes.csv"))
    labels_df = pd.read_csv(os.path.join(data_dir, "labels", f"{split}.csv"))
    id_to_label = {str(r["image_id"]): str(r["label"]) for _, r in labels_df.iterrows()}
    eval_tf = get_transforms(img_size=224, is_train=False)

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []
    h_fwd = model.layer4.register_forward_hook(lambda m, inp, out: activations.append(out))
    h_bwd = model.layer4.register_full_backward_hook(lambda m, gin, gout: gradients.append(gout[0]))

    n = len(image_ids)
    fig, axes = plt.subplots(n, 2, figsize=(8.5, 3.3 * n), dpi=130)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    records = []
    try:
        for i, img_id in enumerate(image_ids):
            true_label = id_to_label[str(img_id)]
            true_idx = class_to_idx[true_label]
            img_path = os.path.join(data_dir, "images", split, f"{img_id}.jpg")
            pil_img = Image.open(img_path).convert("RGB")

            x = eval_tf(pil_img).unsqueeze(0).to(device)
            activations.clear()
            gradients.clear()
            model.zero_grad()

            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            pred_label = idx_to_class[pred_idx]
            conf = float(probs[pred_idx].item())
            is_correct = pred_idx == true_idx

            if expected_correct is not None and is_correct != expected_correct:
                raise ValueError(
                    f"Imagem {img_id}: esperado correct={expected_correct}, mas obteve "
                    f"Real='{true_label}' vs Pred='{pred_label}' (correct={is_correct})"
                )

            logits[0, pred_idx].backward()
            act = activations[0][0].detach().cpu().numpy()
            grad = gradients[0][0].detach().cpu().numpy()
            weights = np.mean(grad, axis=(1, 2))
            cam = np.maximum(np.tensordot(weights, act, axes=(0, 0)), 0)
            if cam.max() > 0:
                cam = cam / cam.max()

            disp_img = np.asarray(pil_img.resize((224, 224)), dtype=np.float32) / 255.0
            cam_resized = np.asarray(
                Image.fromarray(np.uint8(cam * 255)).resize((224, 224), resample=Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            heatmap = mpl_cm.jet(cam_resized)[:, :, :3]
            overlay = np.clip(0.55 * disp_img + 0.45 * heatmap, 0.0, 1.0)

            axes[i, 0].imshow(disp_img)
            axes[i, 0].set_title(f"ID: {img_id} | Real: {true_label}", fontsize=9.5, fontweight="bold")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(
                f"Grad-CAM | Pred: {pred_label} ({conf * 100:.1f}%)",
                fontsize=9.5,
                fontweight="bold",
            )
            axes[i, 1].axis("off")

            records.append({
                "image_id": img_id,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": round(conf, 4),
                "correct": is_correct,
            })
    finally:
        h_fwd.remove()
        h_bwd.remove()

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, pd.DataFrame(records)

