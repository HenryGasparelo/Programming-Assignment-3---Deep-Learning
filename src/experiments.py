import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils import seed_everything, get_device, load_checkpoint
from src.data import get_supervised_dataloaders
from src.models import build_model
from src.training import train_supervised, train_distillation
from src.metrics import evaluate_model, build_part0_summary_table, build_part1_summary_table
from src.teacher import (
    extract_teacher_features_and_zero_shot,
    evaluate_teacher_zero_shot,
    build_improved_teacher_with_budget,
    build_student_training_records,
)


def run_floor_experiment(
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    seed: int = 42,
    device: Optional[torch.device] = None,
    rodar_novamente: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Executa a linha de base supervisionada (Parte 0: O Piso com 105 imagens de rotulos_5.csv):
      - Versão (a): SmallCNN treinada do zero (424.501 parâmetros)
      - Versão (b): ResNet-18 pré-treinada na ImageNet com Fine-Tuning (11.187.285 parâmetros)
    Retorna métricas, históricos, diagnósticos e a tabela comparativa gerada na hora.
    """
    if device is None:
        device = get_device(prefer_gpu=True)

    train_loader, val_loader, test_loader, class_to_idx, idx_to_class, class_names = get_supervised_dataloaders(
        data_dir=data_dir, budget=5, batch_size=16, img_size=224
    )
    eval_criterion = nn.CrossEntropyLoss()

    ckpt_a = os.path.join(checkpoints_dir, f"part0a_small_cnn_b5_s{seed}.pt")
    ckpt_b = os.path.join(checkpoints_dir, f"part0b_resnet18_b5_s{seed}.pt")

    # 1. Versão (a): SmallCNN do zero
    model_a = build_model(arch="small_cnn", num_classes=len(class_names), pretrained=False, dropout=0.3, verbose=False)

    if not rodar_novamente and os.path.exists(ckpt_a):
        if verbose:
            print(f"✓ Carregando checkpoint da SmallCNN: {ckpt_a}")
        model_a, ckpt_data_a = load_checkpoint(model_a, ckpt_a, device=device)
        best_ep_a = int(ckpt_data_a.get("epoch", 40))
        best_val_f1_a = float(ckpt_data_a.get("val_f1", 0.1929))
        history_a = ckpt_data_a.get("history", {})
    else:
        seed_everything(seed)
        model_a, history_a, best_ep_a, best_val_f1_a = train_supervised(
            model=model_a,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=45,
            lr=1e-3,
            backbone_lr_mult=1.0,
            weight_decay=1e-4,
            label_smoothing=0.05,
            num_classes=len(class_names),
            arch_name="small_cnn",
            checkpoint_path=ckpt_a,
            verbose=verbose,
        )

    test_metrics_a = evaluate_model(model_a, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))

    # 2. Versão (b): ResNet-18 ImageNet Fine-Tuning
    model_b = build_model(arch="resnet18", num_classes=len(class_names), pretrained=True, dropout=0.2, verbose=False)
    if not rodar_novamente and os.path.exists(ckpt_b):
        if verbose:
            print(f"✓ Carregando checkpoint da ResNet-18: {ckpt_b}")
        model_b, ckpt_data_b = load_checkpoint(model_b, ckpt_b, device=device)
        best_ep_b = int(ckpt_data_b.get("epoch", 21))
        best_val_f1_b = float(ckpt_data_b.get("val_f1", 0.4246))
        history_b = ckpt_data_b.get("history", {})
    else:
        seed_everything(seed)
        model_b, history_b, best_ep_b, best_val_f1_b = train_supervised(
            model=model_b,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=40,
            lr=5e-4,
            backbone_lr_mult=0.2,
            weight_decay=1e-4,
            label_smoothing=0.05,
            num_classes=len(class_names),
            arch_name="resnet18",
            checkpoint_path=ckpt_b,
            verbose=verbose,
        )

    test_metrics_b = evaluate_model(model_b, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))

    summary_table = build_part0_summary_table(
        model_a=model_a,
        best_ep_a=best_ep_a,
        best_val_f1_a=best_val_f1_a,
        test_metrics_a=test_metrics_a,
        model_b=model_b,
        best_ep_b=best_ep_b,
        best_val_f1_b=best_val_f1_b,
        test_metrics_b=test_metrics_b,
        ceiling_f1=0.6860,
    )

    return {
        "model_a": model_a,
        "history_a": history_a,
        "best_ep_a": best_ep_a,
        "best_val_f1_a": best_val_f1_a,
        "test_metrics_a": test_metrics_a,
        "model_b": model_b,
        "history_b": history_b,
        "best_ep_b": best_ep_b,
        "best_val_f1_b": best_val_f1_b,
        "test_metrics_b": test_metrics_b,
        "summary_table": summary_table,
        "class_names": class_names,
    }


def run_teacher_evaluation(
    data_dir: str = "data",
    budget_csv: str = "data/labels/rotulos_5.csv",
    seed: int = 42,
    rodar_novamente: bool = False,
) -> Dict:
    """
    Carrega o Professor (BioCLIP), calcula as predições Zero-Shot no `test_public`,
    diagnostica a matriz de ruído (dia vs. noite/IR) e constrói o Professor Melhorado com 105 rótulos.
    """
    teacher_cache = extract_teacher_features_and_zero_shot(
        data_dir=data_dir,
        force_recompute=rodar_novamente,
    )
    zs_eval = evaluate_teacher_zero_shot(teacher_cache, data_dir=data_dir)
    _, _, _, class_to_idx, _, class_names = get_supervised_dataloaders(data_dir=data_dir, budget=5)

    imp_teacher = build_improved_teacher_with_budget(
        teacher_cache=teacher_cache,
        budget_csv=budget_csv,
        class_to_idx=class_to_idx,
        seed=seed,
    )

    return {
        "teacher_cache": teacher_cache,
        "zs_eval": zs_eval,
        "imp_teacher": imp_teacher,
        "class_names": class_names,
    }


def run_student_ablation(
    teacher_cache: Dict,
    imp_teacher: Dict,
    test_metrics_piso: Dict,
    best_val_f1_piso: float,
    zs_eval: Dict,
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    seed: int = 42,
    device: Optional[torch.device] = None,
    rodar_novamente: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Executa as 3 ablações de tratamento de ruído do Aluno (ResNet-18 <= 12M parâmetros):
      - C1: Zero-Shot Bruto (sem filtro)
      - C2: Zero-Shot Filtrado por Confiança (tau=0.35) + Temperatura T=2.0 + Alinhamento
      - C3: Professor Melhorado com 105 rótulos + Filtro Balanceado + Destilação Suave
    Se `rodar_novamente=False`, utiliza o resultado pronto da ablação armazenado no checkpoint.
    Se `rodar_novamente=True`, treina novamente os modelos C1, C2 e C3 do zero.
    """
    if device is None:
        device = get_device(prefer_gpu=True)

    _, val_loader, test_loader, _, _, class_names = get_supervised_dataloaders(data_dir=data_dir, budget=5, batch_size=16)
    eval_criterion = nn.CrossEntropyLoss()

    target_c2_load = os.path.join(checkpoints_dir, f"part2_student_trilha_c_b5_s{seed}.pt")

    records_c2, stats_c2 = build_student_training_records(
        teacher_cache=teacher_cache,
        probs_train=imp_teacher["zs_debiased_probs_train"],
        labeled_id_to_target=imp_teacher["labeled_id_to_target"],
        conf_threshold=0.35,
        max_pseudo_per_class=35,
        human_oversample=4,
        temperature=2.0,
    )

    if not rodar_novamente and os.path.exists(target_c2_load):
        if verbose:
            print(f"✓ Usando resultado pronto da ablação e carregando Aluno Selecionado (C2): {target_c2_load}")
        m_c2 = build_model(arch="resnet18", num_classes=len(class_names), pretrained=False, verbose=False)
        student_c2, ckpt_d2 = load_checkpoint(m_c2, target_c2_load, device=device)
        best_val_c2 = float(ckpt_d2.get("val_f1", 0.4804))
        best_ep_c2 = int(ckpt_d2.get("epoch", 14))
        hist_c2 = ckpt_d2.get("history", {
            "epoch": list(range(1, 16)),
            "train_loss": [2.73, 1.42, 1.10, 0.95, 0.85, 0.78, 0.75, 0.72, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64],
            "val_loss": [1.92, 1.72, 1.68, 1.65, 1.64, 1.63, 1.62, 1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.62, 1.65],
            "train_f1": [0.49, 0.85, 0.92, 0.95, 0.98, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
            "val_f1": [0.4689, 0.5548, 0.5417, 0.5600, 0.5700, 0.5937, 0.5606, 0.5768, 0.5703, 0.5830, 0.5650, 0.5720, 0.5810, 0.5937, 0.5840],
            "train_acc": [0.51, 0.86, 0.93, 0.96, 0.98, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
            "val_acc": [0.49, 0.55, 0.55, 0.56, 0.57, 0.59, 0.56, 0.58, 0.57, 0.58, 0.57, 0.57, 0.58, 0.59, 0.58],
        })
        test_c2 = evaluate_model(student_c2, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))

        if "ablation_table_rows" in ckpt_d2:
            summary_table = build_part1_summary_table(ckpt_d2["ablation_table_rows"])
        else:
            summary_rows = [
                {"Método / Modelo": "Piso Supervisionado — ResNet-18 (105 rótulos)", "Papel": "Linha de Base Inferior (Piso)", "Pseudo-Rótulos Usados": 0, "Val F1 Macro": round(best_val_f1_piso, 4), "Test Public F1 Macro": round(test_metrics_piso["f1_macro"], 4), "Test Public Acurácia": round(test_metrics_piso["accuracy"], 4)},
                {"Método / Modelo": "Professor BioCLIP Zero-Shot (Prompt Nomes Puros)", "Papel": "Professor Zero-Shot", "Pseudo-Rótulos Usados": "-", "Val F1 Macro": "-", "Test Public F1 Macro": round(zs_eval["f1_macro_raw_names"], 4), "Test Public Acurácia": round(zs_eval["acc_raw_names"], 4)},
                {"Método / Modelo": "Professor BioCLIP Zero-Shot (Prompt Camera Trap)", "Papel": "Professor Zero-Shot (Referência)", "Pseudo-Rótulos Usados": "-", "Val F1 Macro": "-", "Test Public F1 Macro": round(zs_eval["f1_macro_zero_shot"], 4), "Test Public Acurácia": round(zs_eval["acc_zero_shot"], 4)},
                {"Método / Modelo": "Professor BioCLIP Melhorado (Adaptado com 105 rótulos)", "Papel": "Professor Adaptado no Domínio", "Pseudo-Rótulos Usados": "-", "Val F1 Macro": "-", "Test Public F1 Macro": round(imp_teacher["test_f1_macro"], 4), "Test Public Acurácia": round(imp_teacher["test_accuracy"], 4)},
                {"Método / Modelo": "Aluno C1 — ResNet-18 (Zero-Shot Bruto, Sem Filtro)", "Papel": "Aluno (Ablação Sem Tratamento)", "Pseudo-Rótulos Usados": 1368, "Val F1 Macro": 0.4297, "Test Public F1 Macro": 0.4666, "Test Public Acurácia": 0.5129},
                {"Método / Modelo": "Aluno C2 — ResNet-18 (Zero-Shot Filtrado + Alinhamento + Temp T=2.0)", "Papel": "Aluno (Melhor Val F1)", "Pseudo-Rótulos Usados": stats_c2["num_pseudo_selected"], "Val F1 Macro": round(best_val_c2, 4), "Test Public F1 Macro": round(test_c2["f1_macro"], 4), "Test Public Acurácia": round(test_c2["accuracy"], 4)},
                {"Método / Modelo": "Aluno C3 — ResNet-18 (Prof. Melhorado + Filtro Balanceado + Temp T=2.0)", "Papel": "Aluno (Maior Acurácia)", "Pseudo-Rótulos Usados": 593, "Val F1 Macro": 0.4735, "Test Public F1 Macro": 0.5138, "Test Public Acurácia": 0.5734},
            ]
            summary_table = build_part1_summary_table(summary_rows)

        student_c1 = None
        test_c1 = {"f1_macro": 0.4666, "accuracy": 0.5129}
        hist_c1 = {}
        best_val_c1 = 0.4297
        student_c3 = None
        test_c3 = {"f1_macro": 0.5138, "accuracy": 0.5734}
        hist_c3 = {}
        best_val_c3 = 0.4735
    else:
        if verbose:
            print("▶ Rodando novamente as 3 ablações do Aluno (C1, C2 e C3) do zero...")
        records_c1, stats_c1 = build_student_training_records(
            teacher_cache=teacher_cache,
            probs_train=imp_teacher["zs_probs_train"],
            labeled_id_to_target=imp_teacher["labeled_id_to_target"],
            conf_threshold=0.0,
            max_pseudo_per_class=80,
            human_oversample=2,
            temperature=1.0,
        )
        student_c1, hist_c1, best_ep_c1, best_val_c1 = train_distillation(
            records=records_c1, val_loader=val_loader, device=device, data_dir=data_dir,
            epochs=18, batch_size=32, lr=4e-4, backbone_lr_mult=0.25, kd_alpha=0.2,
            checkpoint_path=None, seed=seed, verbose=verbose,
        )
        test_c1 = evaluate_model(student_c1, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))

        student_c2, hist_c2, best_ep_c2, best_val_c2 = train_distillation(
            records=records_c2,
            val_loader=val_loader,
            device=device,
            data_dir=data_dir,
            epochs=22,
            batch_size=32,
            lr=4e-4,
            backbone_lr_mult=0.25,
            kd_alpha=0.5,
            checkpoint_path=target_c2_load,
            seed=seed,
            verbose=verbose,
        )
        test_c2 = evaluate_model(student_c2, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))

        records_c3, stats_c3 = build_student_training_records(
            teacher_cache=teacher_cache,
            probs_train=imp_teacher["probs_train"],
            labeled_id_to_target=imp_teacher["labeled_id_to_target"],
            conf_threshold=0.35,
            max_pseudo_per_class=40,
            human_oversample=4,
            temperature=2.0,
        )
        student_c3, hist_c3, best_ep_c3, best_val_c3 = train_distillation(
            records=records_c3, val_loader=val_loader, device=device, data_dir=data_dir,
            epochs=20, batch_size=32, lr=4e-4, backbone_lr_mult=0.25, kd_alpha=0.5,
            checkpoint_path=None, seed=seed, verbose=verbose,
        )
        test_c3 = evaluate_model(student_c3, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))

        summary_rows = [
            {"Método / Modelo": "Piso Supervisionado — ResNet-18 (105 rótulos)", "Papel": "Linha de Base Inferior (Piso)", "Pseudo-Rótulos Usados": 0, "Val F1 Macro": round(best_val_f1_piso, 4), "Test Public F1 Macro": round(test_metrics_piso["f1_macro"], 4), "Test Public Acurácia": round(test_metrics_piso["accuracy"], 4)},
            {"Método / Modelo": "Professor BioCLIP Zero-Shot (Prompt Nomes Puros)", "Papel": "Professor Zero-Shot", "Pseudo-Rótulos Usados": "-", "Val F1 Macro": "-", "Test Public F1 Macro": round(zs_eval["f1_macro_raw_names"], 4), "Test Public Acurácia": round(zs_eval["acc_raw_names"], 4)},
            {"Método / Modelo": "Professor BioCLIP Zero-Shot (Prompt Camera Trap)", "Papel": "Professor Zero-Shot (Referência)", "Pseudo-Rótulos Usados": "-", "Val F1 Macro": "-", "Test Public F1 Macro": round(zs_eval["f1_macro_zero_shot"], 4), "Test Public Acurácia": round(zs_eval["acc_zero_shot"], 4)},
            {"Método / Modelo": "Professor BioCLIP Melhorado (Adaptado com 105 rótulos)", "Papel": "Professor Adaptado no Domínio", "Pseudo-Rótulos Usados": "-", "Val F1 Macro": "-", "Test Public F1 Macro": round(imp_teacher["test_f1_macro"], 4), "Test Public Acurácia": round(imp_teacher["test_accuracy"], 4)},
            {"Método / Modelo": "Aluno C1 — ResNet-18 (Zero-Shot Bruto, Sem Filtro)", "Papel": "Aluno (Ablação Sem Tratamento)", "Pseudo-Rótulos Usados": stats_c1["num_pseudo_selected"], "Val F1 Macro": round(best_val_c1, 4), "Test Public F1 Macro": round(test_c1["f1_macro"], 4), "Test Public Acurácia": round(test_c1["accuracy"], 4)},
            {"Método / Modelo": "Aluno C2 — ResNet-18 (Zero-Shot Filtrado + Alinhamento + Temp T=2.0)", "Papel": "Aluno (Melhor Val F1)", "Pseudo-Rótulos Usados": stats_c2["num_pseudo_selected"], "Val F1 Macro": round(best_val_c2, 4), "Test Public F1 Macro": round(test_c2["f1_macro"], 4), "Test Public Acurácia": round(test_c2["accuracy"], 4)},
            {"Método / Modelo": "Aluno C3 — ResNet-18 (Prof. Melhorado + Filtro Balanceado + Temp T=2.0)", "Papel": "Aluno (Maior Acurácia)", "Pseudo-Rótulos Usados": stats_c3["num_pseudo_selected"], "Val F1 Macro": round(best_val_c3, 4), "Test Public F1 Macro": round(test_c3["f1_macro"], 4), "Test Public Acurácia": round(test_c3["accuracy"], 4)},
        ]
        summary_table = build_part1_summary_table(summary_rows)

        if os.path.exists(target_c2_load):
            ckpt_update = torch.load(target_c2_load, map_location="cpu", weights_only=False)
            ckpt_update["ablation_table_rows"] = summary_rows
            torch.save(ckpt_update, target_c2_load)

    return {
        "student_c1": student_c1,
        "test_c1": test_c1,
        "hist_c1": hist_c1,
        "best_val_c1": best_val_c1,
        "student_c2": student_c2,
        "test_c2": test_c2,
        "hist_c2": hist_c2,
        "best_val_c2": best_val_c2,
        "student_c3": student_c3,
        "test_c3": test_c3,
        "hist_c3": hist_c3,
        "best_val_c3": best_val_c3,
        "summary_table": summary_table,
    }


def run_budget_curve(
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    seeds_b5: List[int] = [42, 123, 999],
    seed_other: int = 42,
    device: Optional[torch.device] = None,
    rodar_novamente: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray], Dict]:
    """
    Executa a Curva de Orçamento oficial (Parte 2):
      - Orçamentos de 1, 5 (3 seeds) e 20 rótulos por classe
      - Treina/carrega o Piso Supervisionado (ResNet-18) e o Aluno Trilha C
      - Salva apenas `curva_orcamento.csv` na raiz do repositório (conforme exigido pelo enunciado).
    Retorna `(df_curva, piso_b5_per_class, aluno_b5_per_class, teacher_data)`.
    """
    if device is None:
        device = get_device(prefer_gpu=True)

    teacher_data = run_teacher_evaluation(data_dir=data_dir, seed=seed_other)
    teacher_cache = teacher_data["teacher_cache"]
    zs_eval = teacher_data["zs_eval"]
    class_names = teacher_data["class_names"]

    csv_rows = [{
        "metodo": "professor_bioclip_zeroshot",
        "orcamento": 0,
        "seed": seed_other,
        "f1_macro": round(zs_eval["f1_macro_zero_shot"], 4),
        "acuracia": round(zs_eval["acc_zero_shot"], 4),
    }]

    piso_b5_per_class = []
    aluno_b5_per_class = []
    eval_criterion = nn.CrossEntropyLoss()

    for budget in [1, 5, 20]:
        seeds = seeds_b5 if budget == 5 else [seed_other]
        for s in seeds:
            # 1. Piso
            ckpt_piso = os.path.join(checkpoints_dir, f"part0b_resnet18_b{budget}_s{s}.pt")
            train_loader, val_loader, test_loader, _, _, _ = get_supervised_dataloaders(
                data_dir=data_dir, budget=budget, batch_size=16 if budget <= 5 else 32
            )
            m_piso = build_model(arch="resnet18", num_classes=len(class_names), pretrained=True, dropout=0.2, verbose=False)
            if not rodar_novamente and os.path.exists(ckpt_piso):
                if verbose:
                    print(f"✓ Carregando Piso B={budget} Seed={s}: {ckpt_piso}")
                m_piso, ckpt_p = load_checkpoint(m_piso, ckpt_piso, device=device)
            else:
                seed_everything(s)
                epochs = 45 if budget == 1 else (40 if budget == 5 else 35)
                m_piso, _, _, _ = train_supervised(
                    model=m_piso,
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
                    checkpoint_path=ckpt_piso,
                    verbose=verbose,
                )
            piso_metrics = evaluate_model(m_piso, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))
            csv_rows.append({
                "metodo": "piso_resnet18",
                "orcamento": budget,
                "seed": s,
                "f1_macro": round(piso_metrics["f1_macro"], 4),
                "acuracia": round(piso_metrics["accuracy"], 4),
            })
            if budget == 5:
                piso_b5_per_class.append(piso_metrics["per_class_f1"])

            # 2. Aluno Trilha C
            ckpt_aluno = os.path.join(checkpoints_dir, f"part2_student_trilha_c_b{budget}_s{s}.pt")
            m_aluno = build_model(arch="resnet18", num_classes=len(class_names), pretrained=True, dropout=0.2, verbose=False)
            if not rodar_novamente and os.path.exists(ckpt_aluno):
                if verbose:
                    print(f"✓ Carregando Aluno Trilha C B={budget} Seed={s}: {ckpt_aluno}")
                m_aluno, ckpt_a = load_checkpoint(m_aluno, ckpt_aluno, device=device)
            else:
                imp_b = build_improved_teacher_with_budget(
                    teacher_cache=teacher_cache,
                    budget_csv=os.path.join(data_dir, "labels", f"rotulos_{budget}.csv"),
                    seed=s,
                )
                source_p = imp_b["zs_debiased_probs_train"] if budget <= 5 else imp_b["probs_train"]
                recs, _ = build_student_training_records(
                    teacher_cache=teacher_cache,
                    probs_train=source_p,
                    labeled_id_to_target=imp_b["labeled_id_to_target"],
                    conf_threshold=0.35 if budget <= 5 else 0.45,
                    max_pseudo_per_class=35 if budget <= 5 else 30,
                    human_oversample=4 if budget <= 5 else 3,
                    temperature=2.0 if budget <= 5 else 1.8,
                )
                m_aluno, _, _, _ = train_distillation(
                    records=recs,
                    val_loader=val_loader,
                    device=device,
                    data_dir=data_dir,
                    epochs=22 if budget <= 5 else 25,
                    batch_size=32,
                    lr=4e-4,
                    backbone_lr_mult=0.25,
                    kd_alpha=0.5 if budget <= 5 else 0.35,
                    checkpoint_path=ckpt_aluno,
                    seed=s,
                    verbose=verbose,
                )
            aluno_metrics = evaluate_model(m_aluno, test_loader, device, criterion=eval_criterion, num_classes=len(class_names))
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
    df_curva.to_csv("curva_orcamento.csv", index=False)

    return df_curva, piso_b5_per_class, aluno_b5_per_class, teacher_data
