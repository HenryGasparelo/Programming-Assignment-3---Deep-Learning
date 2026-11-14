import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.utils import seed_everything, get_device, save_json
from src.data import get_supervised_dataloaders
from src.models import build_model
from src.training import train_supervised
from src.metrics import evaluate_model
from src.unlabeled import (
    extract_teacher_features_and_zero_shot,
    evaluate_teacher_zero_shot,
    build_improved_teacher_with_budget,
    build_student_training_records,
    train_student_trilha_c,
)


def get_trilha_c_hyperparams_for_budget(budget: int) -> Dict:
    """
    Retorna os hiperparâmetros de filtragem e adaptação da Trilha C para cada orçamento (1, 5 ou 20 rótulos/classe).
    - Em B=1 (21 rótulos): apoia-se no Zero-Shot Desenviesado + Protótipos visuais (1 exemplo/classe).
    - Em B=5 (105 rótulos): usa a configuração validada na Parte 1 (Zero-Shot Desenviesado + Filtro tau=0.35 + Temp=2.0).
    - Em B=20 (420 rótulos): usa o Professor Melhorado com limiar mais estrito (tau=0.45) para adicionar apenas
      pseudo-rótulos de altíssima confiança às 420 imagens humanas.
    """
    if budget == 1:
        return {
            "use_debiased_zs": False,
            "w_zs": 0.65,
            "w_proto": 0.35,
            "w_lr": 0.00,
            "conf_threshold": 0.32,
            "max_pseudo_per_class": 35,
            "human_oversample": 8,
            "temperature": 2.0,
            "epochs": 20,
            "batch_size": 32,
            "lr": 4e-4,
            "kd_alpha": 0.5,
        }
    elif budget == 5:
        return {
            "use_debiased_zs": True,
            "w_zs": 0.55,
            "w_proto": 0.25,
            "w_lr": 0.20,
            "conf_threshold": 0.35,
            "max_pseudo_per_class": 35,
            "human_oversample": 4,
            "temperature": 2.0,
            "epochs": 22,
            "batch_size": 32,
            "lr": 4e-4,
            "kd_alpha": 0.5,
        }
    elif budget == 20:
        return {
            "use_debiased_zs": False,
            "w_zs": 0.30,
            "w_proto": 0.25,
            "w_lr": 0.45,
            "conf_threshold": 0.45,
            "max_pseudo_per_class": 30,
            "human_oversample": 3,
            "temperature": 1.8,
            "epochs": 25,
            "batch_size": 32,
            "lr": 4e-4,
            "kd_alpha": 0.35,
        }
    else:
        raise ValueError(f"Orçamento não suportado: {budget}")


def train_or_load_piso(
    budget: int,
    seed: int,
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    device: Optional[torch.device] = None,
    reuse_checkpoint: bool = True,
    verbose: bool = True,
) -> Tuple[Dict, float, int]:
    """
    Treina (ou carrega do checkpoint se já existir) o Piso Supervisionado (Parte 0b: ResNet-18 ImageNet)
    para um dado orçamento (`budget` in {1, 5, 20}) e `seed`.
    """
    if device is None:
        device = get_device(prefer_gpu=True)

    ckpt_path = os.path.join(checkpoints_dir, f"part0b_resnet18_b{budget}_s{seed}.pt")
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class, class_names = get_supervised_dataloaders(
        data_dir=data_dir, budget=budget, batch_size=16 if budget <= 5 else 32, img_size=224
    )
    eval_criterion = nn.CrossEntropyLoss()

    if reuse_checkpoint and os.path.exists(ckpt_path):
        if verbose:
            print(f"✓ [Piso Supervisionado | B={budget} | Seed={seed}] Carregando checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(arch="resnet18", num_classes=len(class_names), pretrained=False, dropout=0.2, verbose=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_f1 = float(ckpt.get("val_f1", 0.0))
        best_ep = int(ckpt.get("epoch", 0))
    else:
        print(f"\n--- [Piso Supervisionado | Orçamento={budget}/classe ({budget*21} imgs) | Seed={seed}] ---")
        seed_everything(seed)
        train_loader, val_loader, test_loader, class_to_idx, idx_to_class, class_names = get_supervised_dataloaders(
            data_dir=data_dir, budget=budget, batch_size=16 if budget <= 5 else 32, img_size=224
        )
        epochs = 45 if budget == 1 else (40 if budget == 5 else 35)
        model = build_model(arch="resnet18", num_classes=len(class_names), pretrained=True, dropout=0.2, verbose=False)
        model, _, best_ep, best_val_f1 = train_supervised(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=5e-4,
            backbone_lr_mult=0.2,
            weight_decay=1e-4,
            label_smoothing=0.05,
            num_classes=len(class_names),
            arch_name="resnet18",
            checkpoint_path=ckpt_path,
            verbose=verbose,
        )

    test_metrics = evaluate_model(model, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))
    if verbose:
        print(
            f"-> [Piso B={budget} | Seed={seed}] Val F1: {best_val_f1:.4f} (Ep {best_ep}) | "
            f"Test Public F1 Macro: {test_metrics['f1_macro']:.4f} | Acc: {test_metrics['accuracy']:.4f}"
        )
    return test_metrics, best_val_f1, best_ep


def train_or_load_student_trilha_c(
    budget: int,
    seed: int,
    teacher_cache: Dict,
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    device: Optional[torch.device] = None,
    reuse_checkpoint: bool = True,
    verbose: bool = True,
) -> Tuple[Dict, float, int, Dict]:
    """
    Treina (ou carrega do checkpoint se já existir) o Aluno da Trilha C (`ResNet-18`)
    para um dado orçamento (`budget` in {1, 5, 20}) e `seed`.
    """
    if device is None:
        device = get_device(prefer_gpu=True)

    # No orçamento 5 e seed 42, podemos reutilizar o checkpoint validado na Parte 1 (part1_student_c2_filtered_zs_s42.pt)
    default_ckpt = os.path.join(checkpoints_dir, f"part2_student_trilha_c_b{budget}_s{seed}.pt")
    part1_c2_ckpt = os.path.join(checkpoints_dir, f"part1_student_c2_filtered_zs_s{seed}.pt")
    ckpt_to_load = part1_c2_ckpt if (budget == 5 and os.path.exists(part1_c2_ckpt)) else default_ckpt

    _, val_loader, test_loader, class_to_idx, idx_to_class, class_names = get_supervised_dataloaders(
        data_dir=data_dir, budget=budget, batch_size=16, img_size=224
    )
    eval_criterion = nn.CrossEntropyLoss()

    hp = get_trilha_c_hyperparams_for_budget(budget)
    imp_teacher = build_improved_teacher_with_budget(
        teacher_cache=teacher_cache,
        budget_csv=os.path.join(data_dir, "labels", f"rotulos_{budget}.csv"),
        class_to_idx=class_to_idx,
        w_zs=hp["w_zs"],
        w_proto=hp["w_proto"],
        w_lr=hp["w_lr"],
        seed=seed,
    )

    if reuse_checkpoint and os.path.exists(ckpt_to_load):
        if verbose:
            print(f"✓ [Aluno Trilha C | B={budget} | Seed={seed}] Carregando checkpoint: {ckpt_to_load}")
        ckpt = torch.load(ckpt_to_load, map_location=device, weights_only=False)
        model = build_model(arch="resnet18", num_classes=len(class_names), pretrained=False, dropout=0.2, verbose=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_f1 = float(ckpt.get("val_f1", 0.0))
        best_ep = int(ckpt.get("epoch", 0))
    else:
        print(f"\n--- [Aluno Trilha C | Orçamento={budget}/classe ({budget*21} imgs) | Seed={seed}] ---")
        probs_source = imp_teacher["zs_debiased_probs_train"] if hp["use_debiased_zs"] else imp_teacher["probs_train"]
        records, stats = build_student_training_records(
            teacher_cache=teacher_cache,
            probs_train=probs_source,
            labeled_id_to_target=imp_teacher["labeled_id_to_target"],
            conf_threshold=hp["conf_threshold"],
            max_pseudo_per_class=hp["max_pseudo_per_class"],
            human_oversample=hp["human_oversample"],
            temperature=hp["temperature"],
        )
        if verbose:
            print(f"Registros selecionados: {stats['num_pseudo_selected']} pseudo-rótulos + {stats['num_human_oversampled']} humanos")
        model, _, best_ep, best_val_f1 = train_student_trilha_c(
            records=records,
            val_loader=val_loader,
            device=device,
            data_dir=data_dir,
            epochs=hp["epochs"],
            batch_size=hp["batch_size"],
            lr=hp["lr"],
            backbone_lr_mult=0.25,
            kd_alpha=hp["kd_alpha"],
            checkpoint_path=default_ckpt,
            seed=seed,
            verbose=verbose,
        )

    test_metrics = evaluate_model(model, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))
    if verbose:
        print(
            f"-> [Aluno Trilha C B={budget} | Seed={seed}] Val F1: {best_val_f1:.4f} (Ep {best_ep}) | "
            f"Test Public F1 Macro: {test_metrics['f1_macro']:.4f} | Acc: {test_metrics['accuracy']:.4f}"
        )
    return test_metrics, best_val_f1, best_ep, imp_teacher


def plot_budget_curve(
    df_curva: pd.DataFrame,
    teacher_zs_f1: float = 0.3017,
    ceiling_f1: float = 0.6860,
    save_path: Optional[str] = None,
    show: bool = True,
) -> pd.DataFrame:
    """
    Plota o gráfico obrigatório da Parte 2:
    F1 Macro no `test_public` contra o número de rótulos por classe (eixo x em escala log),
    com:
      - Curva do Piso Supervisionado (Parte 0b) + média ± desvio nas 3 seeds do orçamento 5
      - Curva do Aluno Trilha C (ResNet-18) + média ± desvio nas 3 seeds do orçamento 5
      - Linha horizontal do Professor BioCLIP Zero-Shot e curva do Professor Melhorado
      - Linha horizontal do Teto de Referência (0,686 com todos os 6.505 rótulos)
    Retorna também a tabela agregada (média ± desvio por orçamento).
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

    # 1. Linha do Teto Supervisionado (6.505 rótulos)
    ax.axhline(
        ceiling_f1,
        color="#2b2b2b",
        linestyle="--",
        linewidth=2,
        label=f"Teto Supervisionado (6.505 rótulos): F1 = {ceiling_f1:.3f}",
    )

    # 2. Linha do Professor BioCLIP Zero-Shot
    ax.axhline(
        teacher_zs_f1,
        color="#d62728",
        linestyle=":",
        linewidth=2,
        label=f"Professor BioCLIP Zero-Shot (0 rótulos): F1 = {teacher_zs_f1:.3f}",
    )

    # 3. Curva do Piso Supervisionado (Parte 0b)
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

    # 4. Curva do Método Trilha C — Aluno ResNet-18
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

    # Anotações de valores nos pontos
    for i, b in enumerate(budgets):
        p_txt = f"{piso_f1_m[i]:.3f}±{piso_f1_s[i]:.3f}" if piso_f1_s[i] > 0 else f"{piso_f1_m[i]:.3f}"
        a_txt = f"{aluno_f1_m[i]:.3f}±{aluno_f1_s[i]:.3f}" if aluno_f1_s[i] > 0 else f"{aluno_f1_m[i]:.3f}"
        ax.annotate(
            p_txt,
            (b, piso_f1_m[i]),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=9,
            color="#1f77b4",
            fontweight="bold",
        )
        ax.annotate(
            a_txt,
            (b, aluno_f1_m[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9.5,
            color="#1b781b",
            fontweight="bold",
        )

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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Tabela resumo agregada
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
    return pd.DataFrame(summary_rows)


def plot_per_class_gain_budget5(
    class_names: List[str],
    piso_b5_per_class_list: List[np.ndarray],
    aluno_b5_per_class_list: List[np.ndarray],
    data_dir: str = "data",
    save_csv_path: Optional[str] = "results/part2_per_class_gain_b5.csv",
    save_fig_path: Optional[str] = "docs/figures/part2_ganho_por_classe_b5.png",
    show: bool = True,
) -> pd.DataFrame:
    """
    Quebra o ganho por classe no orçamento de 5 rótulos por classe (média das 3 seeds):
    identifica quais classes ganharam mais com os dados não rotulados (Trilha C) e quais não ganharam nada.
    """
    df_test = pd.read_csv(os.path.join(data_dir, "labels", "test_public.csv"))
    support_map = df_test["label"].value_counts().to_dict()

    piso_mat = np.stack(piso_b5_per_class_list, axis=0)   # (3, 21)
    aluno_mat = np.stack(aluno_b5_per_class_list, axis=0) # (3, 21)

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
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        df_gain.to_csv(save_csv_path, index=False)

    df_plot = df_gain.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=130)

    # Subplot 1: Ganho líquido ΔF1 por classe (média de 3 seeds no orçamento 5)
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

    # Subplot 2: Comparação absoluta Piso (média 3 seeds) vs Aluno (média 3 seeds)
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
    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        plt.savefig(save_fig_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return df_gain


def run_part2(
    data_dir: str = "data",
    results_dir: str = "results",
    figures_dir: str = "docs/figures",
    checkpoints_dir: str = "checkpoints",
    seeds_b5: List[int] = [42, 123, 999],
    seed_other: int = 42,
    reuse_checkpoints: bool = True,
    show_plots: bool = False,
) -> Dict:
    """
    Executa na íntegra a Parte 2 — A Curva de Orçamento:
      1. Roda o Piso (Parte 0b) e o Aluno da Trilha C nos orçamentos de 1, 5 (3 seeds) e 20 rótulos/classe.
      2. Salva `results/curva_orcamento.csv` e `curva_orcamento.csv` na raiz (entregável obrigatório).
      3. Plota a curva de orçamento em escala logarítmica com Piso, Aluno, Professor e Teto (0,686).
      4. Quebra o ganho por classe no orçamento de 5 rótulos.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = get_device(prefer_gpu=True)
    teacher_cache = extract_teacher_features_and_zero_shot(data_dir=data_dir)
    zs_eval = evaluate_teacher_zero_shot(teacher_cache, data_dir=data_dir)
    class_names = teacher_cache["class_names"]

    csv_rows = []

    # Registro de referência do Professor Zero-Shot
    csv_rows.append({
        "metodo": "professor_bioclip_zeroshot",
        "orcamento": 0,
        "seed": seed_other,
        "f1_macro": round(zs_eval["f1_macro_zero_shot"], 4),
        "acuracia": round(zs_eval["acc_zero_shot"], 4),
    })

    piso_b5_per_class = []
    aluno_b5_per_class = []

    for budget in [1, 5, 20]:
        seeds = seeds_b5 if budget == 5 else [seed_other]
        for s in seeds:
            # 1) Piso Supervisionado (Parte 0b)
            piso_metrics, piso_val_f1, piso_ep = train_or_load_piso(
                budget=budget,
                seed=s,
                data_dir=data_dir,
                checkpoints_dir=checkpoints_dir,
                device=device,
                reuse_checkpoint=reuse_checkpoints,
                verbose=True,
            )
            csv_rows.append({
                "metodo": "piso_resnet18",
                "orcamento": budget,
                "seed": s,
                "f1_macro": round(piso_metrics["f1_macro"], 4),
                "acuracia": round(piso_metrics["accuracy"], 4),
            })
            if budget == 5:
                piso_b5_per_class.append(piso_metrics["per_class_f1"])

            # 2) Método da Trilha C: Aluno ResNet-18
            aluno_metrics, aluno_val_f1, aluno_ep, imp_teacher = train_or_load_student_trilha_c(
                budget=budget,
                seed=s,
                teacher_cache=teacher_cache,
                data_dir=data_dir,
                checkpoints_dir=checkpoints_dir,
                device=device,
                reuse_checkpoint=reuse_checkpoints,
                verbose=True,
            )
            csv_rows.append({
                "metodo": "trilha_c_aluno_resnet18",
                "orcamento": budget,
                "seed": s,
                "f1_macro": round(aluno_metrics["f1_macro"], 4),
                "acuracia": round(aluno_metrics["accuracy"], 4),
            })
            if budget == 5:
                aluno_b5_per_class.append(aluno_metrics["per_class_f1"])

    df_curva = pd.DataFrame(csv_rows)
    # Salva tanto em results/curva_orcamento.csv quanto na raiz curva_orcamento.csv conforme Seção 4 do enunciado
    df_curva.to_csv(os.path.join(results_dir, "curva_orcamento.csv"), index=False)
    df_curva.to_csv("curva_orcamento.csv", index=False)

    df_agg_table = plot_budget_curve(
        df_curva=df_curva,
        teacher_zs_f1=zs_eval["f1_macro_zero_shot"],
        ceiling_f1=0.6860,
        save_path=os.path.join(figures_dir, "part2_curva_orcamento.png"),
        show=show_plots,
    )
    df_agg_table.to_csv(os.path.join(results_dir, "part2_summary_table.csv"), index=False)

    df_gain_b5 = plot_per_class_gain_budget5(
        class_names=class_names,
        piso_b5_per_class_list=piso_b5_per_class,
        aluno_b5_per_class_list=aluno_b5_per_class,
        data_dir=data_dir,
        save_csv_path=os.path.join(results_dir, "part2_per_class_gain_b5.csv"),
        save_fig_path=os.path.join(figures_dir, "part2_ganho_por_classe_b5.png"),
        show=show_plots,
    )

    return {
        "df_curva": df_curva,
        "df_agg_table": df_agg_table,
        "df_gain_b5": df_gain_b5,
    }


if __name__ == "__main__":
    run_part2(show_plots=False)
