import os
from typing import Dict
import pandas as pd
import torch.nn as nn

from src.utils import seed_everything, get_device, save_json
from src.data import get_supervised_dataloaders
from src.metrics import evaluate_model, plot_confusion_matrix, plot_training_curves
from src.unlabeled import (
    extract_teacher_features_and_zero_shot,
    evaluate_teacher_zero_shot,
    build_improved_teacher_with_budget,
    build_student_training_records,
    train_student_trilha_c,
    plot_teacher_noise_diagnostics,
    plot_student_vs_teacher_per_class,
    build_part1_summary_table,
)


def run_part1(
    data_dir: str = "data",
    results_dir: str = "results",
    figures_dir: str = "docs/figures",
    checkpoints_dir: str = "checkpoints",
    seed: int = 42,
    show_plots: bool = False,
) -> Dict:
    """
    Executa na íntegra a Parte 1 — Trilha C (Professor e Aluno):
      1. Avaliação Zero-Shot do Professor (BioCLIP) no `test_public` e Matriz de Ruído.
      2. Diagnóstico sistemático dos erros do Professor (noite/IR, animais pequenos, classes parecidas).
      3. Ablação de tratamento de ruído no treinamento do Aluno (`ResNet-18`):
         - C1: Aluno com Pseudo-Rótulos Zero-Shot Brutos (sem filtro)
         - C2: Aluno com Filtro de Confiança + Rótulos Suaves com Temperatura sobre Zero-Shot
         - C3: Aluno com Professor Melhorado (105 rótulos) + Filtro Balanceado + Destilação Suave
      4. Comparação por classe: em quais classes o Aluno supera o Professor e por quê.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = get_device(prefer_gpu=True)
    print(f"=== Executando Parte 1 (Trilha C: Professor e Aluno) | Dispositivo: {device} | Seed: {seed} ===\n")

    _, val_loader, test_loader, class_to_idx, idx_to_class, class_names = get_supervised_dataloaders(
        data_dir=data_dir, budget=5, batch_size=16, img_size=224
    )
    eval_criterion = nn.CrossEntropyLoss()

    # 1. Carregar cache do Professor (BioCLIP) e avaliar Zero-Shot
    teacher_cache = extract_teacher_features_and_zero_shot(data_dir=data_dir)
    zs_eval = evaluate_teacher_zero_shot(teacher_cache, data_dir=data_dir)
    print(
        f"-> [Professor BioCLIP Zero-Shot] "
        f"Prompt Nomes Puros F1: {zs_eval['f1_macro_raw_names']:.4f} (Acc: {zs_eval['acc_raw_names']:.4f}) | "
        f"Prompt Camera Trap F1: {zs_eval['f1_macro_zero_shot']:.4f} (Acc: {zs_eval['acc_zero_shot']:.4f})"
    )
    print(
        f"-> [Diagnóstico Dia vs. Noite/IR] "
        f"Frações IR no test_public: {zs_eval['night_fraction']*100:.1f}% | "
        f"Acc Dia: {zs_eval['acc_day']:.4f} | Acc Noite/IR: {zs_eval['acc_night']:.4f}"
    )

    plot_confusion_matrix(
        cm=zs_eval["confusion_matrix"],
        class_names=class_names,
        title=(
            f"Matriz de Confusão (Matriz de Ruído) — Professor BioCLIP Zero-Shot no test_public\n"
            f"(F1 Macro = {zs_eval['f1_macro_zero_shot']:.4f} | Acurácia = {zs_eval['acc_zero_shot']:.4f})"
        ),
        normalize=True,
        save_path=os.path.join(figures_dir, "part1_teacher_zeroshot_confusion_matrix.png"),
        show=show_plots,
    )

    # 2. Construir Professor Melhorado com os 105 rótulos (sem tocar no val!)
    imp_teacher = build_improved_teacher_with_budget(
        teacher_cache=teacher_cache,
        budget_csv=os.path.join(data_dir, "labels", "rotulos_5.csv"),
        class_to_idx=class_to_idx,
        seed=seed,
    )
    print(
        f"-> [Professor Melhorado com 105 rótulos (rotulos_5.csv)] "
        f"Test Public F1 Macro (diagnóstico): {imp_teacher['test_f1_macro']:.4f} | "
        f"Acc: {imp_teacher['test_accuracy']:.4f}\n"
    )

    plot_teacher_noise_diagnostics(
        zs_eval=zs_eval,
        teacher_cache=teacher_cache,
        imp_teacher=imp_teacher,
        save_path=os.path.join(figures_dir, "part1_teacher_noise_diagnostics.png"),
        show=show_plots,
    )

    # 3. Ablação 1 (C1): Aluno treinado nos Pseudo-Rótulos Zero-Shot Brutos (sem filtro) + 105 rótulos
    print("--- [Trilha C - Ablação C1] Treinando Aluno (ResNet-18) com Pseudo-Rótulos Zero-Shot Brutos (Sem Filtro) ---")
    records_c1, stats_c1 = build_student_training_records(
        teacher_cache=teacher_cache,
        probs_train=imp_teacher["zs_probs_train"],
        labeled_id_to_target=imp_teacher["labeled_id_to_target"],
        conf_threshold=0.0,
        max_pseudo_per_class=80,
        human_oversample=2,
        temperature=1.0,
    )
    print(f"Registros de treino C1: {stats_c1['num_pseudo_selected']} pseudo-rótulos + {stats_c1['num_human_oversampled']} humanos")
    student_c1, hist_c1, best_ep_c1, best_val_c1 = train_student_trilha_c(
        records=records_c1,
        val_loader=val_loader,
        device=device,
        data_dir=data_dir,
        epochs=18,
        batch_size=32,
        lr=4e-4,
        backbone_lr_mult=0.25,
        kd_alpha=0.2,
        checkpoint_path=os.path.join(checkpoints_dir, f"part1_student_c1_raw_zs_s{seed}.pt"),
        seed=seed,
        verbose=True,
    )
    test_c1 = evaluate_model(student_c1, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))
    print(f"-> [Resultado Aluno C1 (Zero-Shot Sem Filtro)] Val F1: {best_val_c1:.4f} | Test Public F1: {test_c1['f1_macro']:.4f} | Acc: {test_c1['accuracy']:.4f}\n")

    # 4. Ablação 2 (C2): Aluno com Filtro de Confiança + Temperatura + Alinhamento sobre Zero-Shot
    print("--- [Trilha C - Ablação C2] Treinando Aluno (ResNet-18) com Zero-Shot Filtrado por Confiança + Temperatura ---")
    records_c2, stats_c2 = build_student_training_records(
        teacher_cache=teacher_cache,
        probs_train=imp_teacher["zs_debiased_probs_train"],
        labeled_id_to_target=imp_teacher["labeled_id_to_target"],
        conf_threshold=0.35,
        max_pseudo_per_class=35,
        human_oversample=4,
        temperature=2.0,
    )
    print(f"Registros de treino C2: {stats_c2['num_pseudo_selected']} pseudo-rótulos filtrados + {stats_c2['num_human_oversampled']} humanos")
    student_c2, hist_c2, best_ep_c2, best_val_c2 = train_student_trilha_c(
        records=records_c2,
        val_loader=val_loader,
        device=device,
        data_dir=data_dir,
        epochs=22,
        batch_size=32,
        lr=4e-4,
        backbone_lr_mult=0.25,
        kd_alpha=0.5,
        checkpoint_path=os.path.join(checkpoints_dir, f"part1_student_c2_filtered_zs_s{seed}.pt"),
        seed=seed,
        verbose=True,
    )
    test_c2 = evaluate_model(student_c2, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))
    print(f"-> [Resultado Aluno C2 (Zero-Shot Filtrado + Temp)] Val F1: {best_val_c2:.4f} | Test Public F1: {test_c2['f1_macro']:.4f} | Acc: {test_c2['accuracy']:.4f}\n")

    # 5. Método Principal (C3): Aluno com Professor Melhorado (105 rótulos) + Filtro Balanceado + Destilação Suave
    print("--- [Trilha C - Método Principal C3] Treinando Aluno (ResNet-18) com Professor Melhorado + Filtro Balanceado + Soft KD ---")
    records_c3, stats_c3 = build_student_training_records(
        teacher_cache=teacher_cache,
        probs_train=imp_teacher["probs_train"],
        labeled_id_to_target=imp_teacher["labeled_id_to_target"],
        conf_threshold=0.35,
        max_pseudo_per_class=40,
        human_oversample=4,
        temperature=2.0,
    )
    print(f"Registros de treino C3: {stats_c3['num_pseudo_selected']} pseudo-rótulos de alta confiança + {stats_c3['num_human_oversampled']} humanos")
    student_c3, hist_c3, best_ep_c3, best_val_c3 = train_student_trilha_c(
        records=records_c3,
        val_loader=val_loader,
        device=device,
        data_dir=data_dir,
        epochs=20,
        batch_size=32,
        lr=4e-4,
        backbone_lr_mult=0.25,
        kd_alpha=0.5,
        checkpoint_path=os.path.join(checkpoints_dir, f"part1_student_c3_improved_teacher_b5_s{seed}.pt"),
        seed=seed,
        verbose=True,
    )
    test_c3 = evaluate_model(student_c3, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))
    print(f"-> [Resultado Aluno C3 (Professor Melhorado + Filtro + KD)] Val F1: {best_val_c3:.4f} | Test Public F1: {test_c3['f1_macro']:.4f} | Acc: {test_c3['accuracy']:.4f}\n")

    plot_training_curves(
        history=hist_c3,
        title_prefix="Parte 1 (Trilha C) — Aluno ResNet-18 (Professor Melhorado + Filtro + KD)",
        save_path=os.path.join(figures_dir, "part1_student_c3_curves.png"),
        show=show_plots,
    )

    plot_confusion_matrix(
        cm=test_c3["confusion_matrix"],
        class_names=class_names,
        title=(
            f"Matriz de Confusão — Aluno ResNet-18 (Trilha C) no test_public\n"
            f"(F1 Macro = {test_c3['f1_macro']:.4f} | Acurácia = {test_c3['accuracy']:.4f})"
        ),
        normalize=True,
        save_path=os.path.join(figures_dir, "part1_student_c3_confusion_matrix.png"),
        show=show_plots,
    )

    # Carregar F1 por classe do Piso (Parte 0b) para comparação
    df_p0 = pd.read_csv(os.path.join(results_dir, "part0_per_class_f1.csv"))
    f1_piso = df_p0["f1_part0b_resnet18"].values

    df_per_class_p1 = plot_student_vs_teacher_per_class(
        class_names=class_names,
        f1_piso=f1_piso,
        f1_teacher_zs=zs_eval["per_class_f1_zero_shot"],
        f1_teacher_imp=imp_teacher["test_per_class_f1"],
        f1_student=test_c3["per_class_f1"],
        save_path=os.path.join(figures_dir, "part1_student_vs_teacher_per_class.png"),
        show=show_plots,
    )
    df_per_class_p1.to_csv(os.path.join(results_dir, "part1_per_class_comparison.csv"), index=False)

    summary_rows = [
        {
            "Método / Modelo": "Piso Supervisionado — Parte 0 (b) ResNet-18 (105 rótulos)",
            "Papel": "Linha de Base Inferior (Piso)",
            "Pseudo-Rótulos Usados": 0,
            "Val F1 Macro": 0.4246,
            "Test Public F1 Macro": round(float(f1_piso.mean()), 4),
            "Test Public Acurácia": 0.5068,
        },
        {
            "Método / Modelo": "Professor BioCLIP Zero-Shot (Prompt Nomes Puros)",
            "Papel": "Professor Zero-Shot",
            "Pseudo-Rótulos Usados": "-",
            "Val F1 Macro": "-",
            "Test Public F1 Macro": round(zs_eval["f1_macro_raw_names"], 4),
            "Test Public Acurácia": round(zs_eval["acc_raw_names"], 4),
        },
        {
            "Método / Modelo": "Professor BioCLIP Zero-Shot (Prompt Camera Trap)",
            "Papel": "Professor Zero-Shot (Referência)",
            "Pseudo-Rótulos Usados": "-",
            "Val F1 Macro": "-",
            "Test Public F1 Macro": round(zs_eval["f1_macro_zero_shot"], 4),
            "Test Public Acurácia": round(zs_eval["acc_zero_shot"], 4),
        },
        {
            "Método / Modelo": "Professor BioCLIP Melhorado (Adaptado com 105 rótulos)",
            "Papel": "Professor Adaptado no Domínio",
            "Pseudo-Rótulos Usados": "-",
            "Val F1 Macro": "-",
            "Test Public F1 Macro": round(imp_teacher["test_f1_macro"], 4),
            "Test Public Acurácia": round(imp_teacher["test_accuracy"], 4),
        },
        {
            "Método / Modelo": "Aluno C1 — ResNet-18 (Pseudo-Rótulos Zero-Shot Brutos, Sem Filtro)",
            "Papel": "Aluno (Ablação Sem Tratamento)",
            "Pseudo-Rótulos Usados": stats_c1["num_pseudo_selected"],
            "Val F1 Macro": round(best_val_c1, 4),
            "Test Public F1 Macro": round(test_c1["f1_macro"], 4),
            "Test Public Acurácia": round(test_c1["accuracy"], 4),
        },
        {
            "Método / Modelo": "Aluno C2 — ResNet-18 (Zero-Shot + Filtro Confiança + Temperatura)",
            "Papel": "Aluno (Ablação Filtro + Temp)",
            "Pseudo-Rótulos Usados": stats_c2["num_pseudo_selected"],
            "Val F1 Macro": round(best_val_c2, 4),
            "Test Public F1 Macro": round(test_c2["f1_macro"], 4),
            "Test Public Acurácia": round(test_c2["accuracy"], 4),
        },
        {
            "Método / Modelo": "Aluno C3 — ResNet-18 (Prof. Melhorado + Filtro Balanceado + Soft KD)",
            "Papel": "Aluno Final (Trilha C)",
            "Pseudo-Rótulos Usados": stats_c3["num_pseudo_selected"],
            "Val F1 Macro": round(best_val_c3, 4),
            "Test Public F1 Macro": round(test_c3["f1_macro"], 4),
            "Test Public Acurácia": round(test_c3["accuracy"], 4),
        },
    ]
    df_summary_p1 = build_part1_summary_table(summary_rows)
    df_summary_p1.to_csv(os.path.join(results_dir, "part1_summary_table.csv"), index=False)

    summary_json = {
        "seed": seed,
        "teacher_zero_shot": {
            "f1_macro_raw_names": zs_eval["f1_macro_raw_names"],
            "acc_raw_names": zs_eval["acc_raw_names"],
            "f1_macro_zero_shot": zs_eval["f1_macro_zero_shot"],
            "acc_zero_shot": zs_eval["acc_zero_shot"],
            "night_fraction": zs_eval["night_fraction"],
            "acc_day": zs_eval["acc_day"],
            "acc_night": zs_eval["acc_night"],
        },
        "teacher_improved_105": {
            "test_f1_macro": imp_teacher["test_f1_macro"],
            "test_accuracy": imp_teacher["test_accuracy"],
        },
        "student_c1_raw_zs": {
            "best_val_epoch": best_ep_c1,
            "val_f1_macro": best_val_c1,
            "test_f1_macro": test_c1["f1_macro"],
            "test_accuracy": test_c1["accuracy"],
            "pseudo_used": stats_c1["num_pseudo_selected"],
        },
        "student_c2_filtered_zs": {
            "best_val_epoch": best_ep_c2,
            "val_f1_macro": best_val_c2,
            "test_f1_macro": test_c2["f1_macro"],
            "test_accuracy": test_c2["accuracy"],
            "pseudo_used": stats_c2["num_pseudo_selected"],
        },
        "student_c3_improved_teacher": {
            "best_val_epoch": best_ep_c3,
            "val_f1_macro": best_val_c3,
            "test_f1_macro": test_c3["f1_macro"],
            "test_accuracy": test_c3["accuracy"],
            "pseudo_used": stats_c3["num_pseudo_selected"],
        },
    }
    save_json(summary_json, os.path.join(results_dir, "part1_summary.json"))

    return {
        "summary": summary_json,
        "df_summary": df_summary_p1,
        "df_per_class": df_per_class_p1,
    }


if __name__ == "__main__":
    run_part1(show_plots=False)
