import os
import copy
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from src.utils import seed_everything, save_checkpoint, count_parameters
from src.data import load_class_mapping, get_transforms, CameraTrapDataset
from src.models import build_model
from src.metrics import evaluate_model, plot_confusion_matrix


def format_species_prompts(class_names: List[str], template: str = "a camera trap photo of a {}") -> List[str]:
    """
    Formata os nomes das 21 classes do ENA24 em prompts para avaliação zero-shot do Professor.
    Converte 'american_black_bear' -> 'american black bear'.
    """
    return [template.format(cname.replace("_", " ")) for cname in class_names]


@torch.no_grad()
def extract_teacher_features_and_zero_shot(
    model_id: str = "hf-hub:imageomics/bioclip",
    data_dir: str = "data",
    cache_dir: str = "results/teacher_cache",
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    force_recompute: bool = False,
) -> Dict:
    """
    Carrega o Foundation Model aberto (BioCLIP 'hf-hub:imageomics/bioclip'),
    extrai embeddings visuais e logits zero-shot para `train` (6.505 imagens) e
    `test_public` (811 imagens) — NUNCA olhando o `val` conforme a regra da Trilha C.
    Salva os resultados em disco (`cache_dir`) para execução rápida e reprodutível.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "bioclip_teacher_outputs.pt")

    class_to_idx, idx_to_class, class_names = load_class_mapping(os.path.join(data_dir, "classes.csv"))

    if os.path.exists(cache_file) and not force_recompute:
        print(f"✓ Carregando predições e embeddings do Professor (BioCLIP) em cache: {cache_file}")
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    import open_clip

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Carregando Foundation Model Professor ({model_id}) em {device}...")
    teacher_model, _, preprocess_val = open_clip.create_model_and_transforms(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    prompts_raw = [cname.replace("_", " ") for cname in class_names]
    prompts_photo = format_species_prompts(class_names, template="a photo of a {}")
    prompts_trap = format_species_prompts(class_names, template="a camera trap photo of a {}")

    def encode_text_prompts(prompt_list: List[str]) -> torch.Tensor:
        tokens = tokenizer(prompt_list).to(device)
        text_feats = teacher_model.encode_text(tokens)
        return F.normalize(text_feats, dim=-1)

    text_feats_raw = encode_text_prompts(prompts_raw)
    text_feats_photo = encode_text_prompts(prompts_photo)
    text_feats_trap = encode_text_prompts(prompts_trap)
    text_feats_ensemble = F.normalize((text_feats_raw + text_feats_photo + text_feats_trap) / 3.0, dim=-1)

    logit_scale = teacher_model.logit_scale.exp().item()

    def process_split(df: pd.DataFrame, images_subdir: str, split_name: str) -> Dict:
        ds = CameraTrapDataset(
            df=df,
            images_dir=os.path.join(data_dir, "images", images_subdir),
            class_to_idx=class_to_idx if "label" in df.columns else None,
            transform=preprocess_val,
            return_id=True,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        all_img_feats = []
        all_ids = []
        all_labels = []

        t0 = time.time()
        for step, batch in enumerate(loader, 1):
            if "label" in df.columns:
                imgs, labels, img_ids = batch
                all_labels.append(labels.numpy())
            else:
                imgs, img_ids = batch
            imgs = imgs.to(device)
            img_feats = F.normalize(teacher_model.encode_image(imgs), dim=-1)
            all_img_feats.append(img_feats.cpu())
            all_ids.extend(list(img_ids))
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"  [{split_name}] Batch {step}/{len(loader)} ({time.time() - t0:.1f}s)")

        img_feats_tensor = torch.cat(all_img_feats, dim=0)
        logits_raw = logit_scale * (img_feats_tensor @ text_feats_raw.cpu().T)
        logits_trap = logit_scale * (img_feats_tensor @ text_feats_trap.cpu().T)
        logits_ens = logit_scale * (img_feats_tensor @ text_feats_ensemble.cpu().T)

        out = {
            "image_ids": all_ids,
            "image_features": img_feats_tensor,
            "logits_raw_names": logits_raw,
            "logits_trap_prompt": logits_trap,
            "logits_ensemble": logits_ens,
        }
        if all_labels:
            out["y_true"] = np.concatenate(all_labels)
        return out

    train_img_dir = os.path.join(data_dir, "images", "train")
    train_ids = sorted([f[:-4] for f in os.listdir(train_img_dir) if f.endswith(".jpg")])
    df_train_all = pd.DataFrame({"image_id": train_ids})
    df_test_pub = pd.read_csv(os.path.join(data_dir, "labels", "test_public.csv"))

    test_out = process_split(df_test_pub, "test_public", "test_public")
    train_out = process_split(df_train_all, "train", "train")

    payload = {
        "model_id": model_id,
        "logit_scale": logit_scale,
        "class_names": class_names,
        "text_features_raw": text_feats_raw.cpu(),
        "text_features_trap": text_feats_trap.cpu(),
        "text_features_ensemble": text_feats_ensemble.cpu(),
        "train": train_out,
        "test_public": test_out,
    }
    torch.save(payload, cache_file)
    return payload


def evaluate_teacher_zero_shot(teacher_cache: Dict, data_dir: str = "data") -> Dict:
    """
    Avalia o desempenho Zero-Shot do Professor (BioCLIP) no `test_public` usando os nomes das classes
    como prompt, além de diagnosticar o erro sistemático entre imagens diurnas (coloridas)
    e noturnas (infravermelho/monocromáticas).
    """
    class_names = teacher_cache["class_names"]
    num_classes = len(class_names)
    y_true = teacher_cache["test_public"]["y_true"]

    # 1. Prompt literal apenas com nome da classe ("american black bear")
    logits_raw = teacher_cache["test_public"]["logits_raw_names"]
    preds_raw = logits_raw.argmax(dim=1).numpy()
    f1_raw = float(f1_score(y_true, preds_raw, average="macro", labels=list(range(num_classes)), zero_division=0))
    acc_raw = float(accuracy_score(y_true, preds_raw))

    # 2. Prompt com nome da classe no contexto ("a camera trap photo of a american black bear")
    logits_trap = teacher_cache["test_public"]["logits_trap_prompt"]
    preds_trap = logits_trap.argmax(dim=1).numpy()
    f1_trap = float(f1_score(y_true, preds_trap, average="macro", labels=list(range(num_classes)), zero_division=0))
    acc_trap = float(accuracy_score(y_true, preds_trap))
    per_class_f1_trap = f1_score(y_true, preds_trap, average=None, labels=list(range(num_classes)), zero_division=0)
    cm_trap = confusion_matrix(y_true, preds_trap, labels=list(range(num_classes)))

    # 3. Diagnóstico Dia (Cor) vs. Noite (Infravermelho) no test_public
    is_night = []
    for img_id in teacher_cache["test_public"]["image_ids"]:
        img_path = os.path.join(data_dir, "images", "test_public", f"{img_id}.jpg")
        im = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        rgb_diff = np.abs(im[:, :, 0] - im[:, :, 1]).mean() + np.abs(im[:, :, 1] - im[:, :, 2]).mean()
        is_night.append(rgb_diff < 3.0)
    is_night = np.array(is_night)

    acc_day = float(accuracy_score(y_true[~is_night], preds_trap[~is_night]))
    acc_night = float(accuracy_score(y_true[is_night], preds_trap[is_night]))

    # Pares de maior confusão sistemática na matriz de ruído
    conf_pairs = []
    for i in range(num_classes):
        row_total = int(cm_trap[i].sum())
        for j in range(num_classes):
            if i != j and cm_trap[i, j] > 0:
                conf_pairs.append({
                    "classe_verdadeira": class_names[i],
                    "classe_predita_professor": class_names[j],
                    "erros": int(cm_trap[i, j]),
                    "suporte_classe": row_total,
                    "taxa_confusao": float(cm_trap[i, j] / max(row_total, 1)),
                })
    df_top_confusions = pd.DataFrame(conf_pairs).sort_values("erros", ascending=False).head(12).reset_index(drop=True)

    return {
        "f1_macro_raw_names": f1_raw,
        "acc_raw_names": acc_raw,
        "f1_macro_zero_shot": f1_trap,
        "acc_zero_shot": acc_trap,
        "per_class_f1_zero_shot": per_class_f1_trap,
        "confusion_matrix": cm_trap,
        "y_true": y_true,
        "y_pred_zero_shot": preds_trap,
        "is_night": is_night,
        "night_fraction": float(is_night.mean()),
        "acc_day": acc_day,
        "acc_night": acc_night,
        "df_top_confusions": df_top_confusions,
    }


def build_improved_teacher_with_budget(
    teacher_cache: Dict,
    budget_csv: str = "data/labels/rotulos_5.csv",
    class_to_idx: Optional[Dict[str, int]] = None,
    c_reg: float = 2.0,
    zs_temp: float = 3.5,
    w_zs: float = 0.55,
    w_proto: float = 0.25,
    w_lr: float = 0.20,
    seed: int = 42,
) -> Dict:
    """
    Constrói o 'Professor Melhorado com os Rótulos do Orçamento' (ex: 105 rótulos de rotulos_5.csv),
    sem JAMAIS olhar o conjunto `val`.
    Combina:
      1) Prior Zero-Shot do BioCLIP (com alinhamento de distribuição sobre `train` e temperatura `zs_temp`)
      2) Protótipos visuais de cosseno das amostras rotuladas no espaço do BioCLIP
      3) Adaptador linear regularizado (LogisticRegression L2) ajustado apenas nas amostras rotuladas
    Retorna as probabilidades calibradas para `train` (pseudo-rótulos) e `test_public` (diagnóstico).
    """
    if class_to_idx is None:
        class_to_idx, _, class_names = load_class_mapping("data/classes.csv")
    else:
        class_names = teacher_cache["class_names"]
    num_classes = len(class_names)

    df_labeled = pd.read_csv(budget_csv)
    id_to_idx_train = {str(img_id): i for i, img_id in enumerate(teacher_cache["train"]["image_ids"])}

    labeled_indices = [id_to_idx_train[str(row["image_id"])] for _, row in df_labeled.iterrows()]
    labeled_targets = np.array([class_to_idx[row["label"]] for _, row in df_labeled.iterrows()])
    labeled_id_to_target = {str(row["image_id"]): class_to_idx[row["label"]] for _, row in df_labeled.iterrows()}

    X_train = teacher_cache["train"]["image_features"].numpy()
    X_test = teacher_cache["test_public"]["image_features"].numpy()
    X_labeled = X_train[labeled_indices]

    # 1. Probabilidades Zero-Shot com alinhamento de distribuição marginal no train
    zs_probs_train = torch.softmax(teacher_cache["train"]["logits_trap_prompt"] / zs_temp, dim=1).numpy()
    zs_probs_test = torch.softmax(teacher_cache["test_public"]["logits_trap_prompt"] / zs_temp, dim=1).numpy()

    marginal_prior = np.clip(zs_probs_train.mean(axis=0, keepdims=True), 1e-6, 1.0)
    zs_deb_train = zs_probs_train / (marginal_prior ** 0.5)
    zs_deb_train /= zs_deb_train.sum(axis=1, keepdims=True)

    zs_deb_test = zs_probs_test / (marginal_prior ** 0.5)
    zs_deb_test /= zs_deb_test.sum(axis=1, keepdims=True)

    # 2. Protótipos visuais médios por classe no espaço do BioCLIP
    prototypes = np.zeros((num_classes, X_train.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        mask = labeled_targets == c
        if mask.any():
            p = X_labeled[mask].mean(axis=0)
            prototypes[c] = p / max(np.linalg.norm(p), 1e-8)

    proto_probs_train = torch.softmax(torch.tensor(X_train @ prototypes.T) * 16.0, dim=1).numpy()
    proto_probs_test = torch.softmax(torch.tensor(X_test @ prototypes.T) * 16.0, dim=1).numpy()

    # 3. Adaptador Linear L2 (Logistic Regression) nas amostras do orçamento
    clf = LogisticRegression(C=c_reg, max_iter=1000, random_state=seed)
    clf.fit(X_labeled, labeled_targets)
    lr_probs_train = clf.predict_proba(X_train)
    lr_probs_test = clf.predict_proba(X_test)

    # Combinação ponderada formando o Professor Melhorado
    probs_train = w_zs * zs_deb_train + w_proto * proto_probs_train + w_lr * lr_probs_train
    probs_test = w_zs * zs_deb_test + w_proto * proto_probs_test + w_lr * lr_probs_test

    y_test = teacher_cache["test_public"]["y_true"]
    preds_test = probs_test.argmax(axis=1)
    f1_test = float(f1_score(y_test, preds_test, average="macro", labels=list(range(num_classes)), zero_division=0))
    acc_test = float(accuracy_score(y_test, preds_test))
    per_class_f1_test = f1_score(y_test, preds_test, average=None, labels=list(range(num_classes)), zero_division=0)
    cm_test = confusion_matrix(y_test, preds_test, labels=list(range(num_classes)))

    return {
        "probs_train": probs_train,
        "probs_test": probs_test,
        "zs_probs_train": zs_probs_train,
        "zs_debiased_probs_train": zs_deb_train,
        "labeled_id_to_target": labeled_id_to_target,
        "test_f1_macro": f1_test,
        "test_accuracy": acc_test,
        "test_per_class_f1": per_class_f1_test,
        "test_confusion_matrix": cm_test,
    }


class TeacherStudentDataset(Dataset):
    """
    Dataset híbrido para o treinamento do Aluno (Trilha C):
    Combina exemplos humanos verdadeiros (com peso `true_label_weight`) e exemplos
    pseudo-rotulados pelo Professor (com rótulo hard `pseudo_label` e distribuição suave `soft_probs`).
    """

    def __init__(
        self,
        records: List[Dict],
        images_dir: str,
        transform=None,
    ):
        self.records = records
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_id = rec["image_id"]
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = int(rec["label"])
        soft_probs = torch.tensor(rec["soft_probs"], dtype=torch.float32)
        sample_weight = float(rec["weight"])
        is_human = bool(rec["is_human"])
        return image, target, soft_probs, sample_weight, is_human


def build_student_training_records(
    teacher_cache: Dict,
    probs_train: np.ndarray,
    labeled_id_to_target: Dict[str, int],
    conf_threshold: float = 0.45,
    max_pseudo_per_class: Optional[int] = 45,
    human_oversample: int = 4,
    temperature: float = 1.5,
) -> Tuple[List[Dict], Dict]:
    """
    Seleciona e trata o ruído dos pseudo-rótulos sobre as 6.505 imagens de `train`:
      - Filtra amostras não rotuladas cuja confiança máxima seja `< conf_threshold`.
      - Limita o máximo de pseudo-rótulos por classe (`max_pseudo_per_class`) para impedir que classes
        gigantes (como urso, 690 imagens) esmaguem classes raras (como cavalo, 45 imagens).
      - Gera distribuições suaves temperadas (`temperature`) para destilação de conhecimento.
      - Inclui os rótulos humanos verdadeiros com repetição (`human_oversample`) para ancorar o modelo.
    """
    train_ids = teacher_cache["train"]["image_ids"]
    num_classes = probs_train.shape[1]

    # Suavização por temperatura das probabilidades do professor
    log_p = np.log(np.clip(probs_train, 1e-8, 1.0)) / temperature
    log_p -= log_p.max(axis=1, keepdims=True)
    soft_probs_all = np.exp(log_p)
    soft_probs_all /= soft_probs_all.sum(axis=1, keepdims=True)

    human_records = []
    candidates_by_class = {c: [] for c in range(num_classes)}

    for idx, img_id in enumerate(train_ids):
        img_id_str = str(img_id)
        if img_id_str in labeled_id_to_target:
            true_c = labeled_id_to_target[img_id_str]
            one_hot = np.zeros(num_classes, dtype=np.float32)
            one_hot[true_c] = 1.0
            # Mistura leve do one-hot humano com o conhecimento estrutural do professor
            blended_soft = 0.85 * one_hot + 0.15 * soft_probs_all[idx]
            rec = {
                "image_id": img_id_str,
                "label": true_c,
                "soft_probs": blended_soft,
                "weight": 1.5,
                "confidence": 1.0,
                "is_human": True,
            }
            for _ in range(human_oversample):
                human_records.append(rec)
        else:
            pred_c = int(probs_train[idx].argmax())
            conf = float(probs_train[idx, pred_c])
            if conf >= conf_threshold:
                candidates_by_class[pred_c].append({
                    "image_id": img_id_str,
                    "label": pred_c,
                    "soft_probs": soft_probs_all[idx],
                    "weight": float(np.clip(conf, 0.4, 1.0)),
                    "confidence": conf,
                    "is_human": False,
                })

    selected_pseudo = []
    per_class_counts = {}
    for c in range(num_classes):
        cands = candidates_by_class[c]
        cands.sort(key=lambda r: r["confidence"], reverse=True)
        if max_pseudo_per_class is not None:
            cands = cands[:max_pseudo_per_class]
        selected_pseudo.extend(cands)
        per_class_counts[c] = len(cands)

    all_records = human_records + selected_pseudo
    stats = {
        "num_human_unique": len(labeled_id_to_target),
        "num_human_oversampled": len(human_records),
        "num_pseudo_selected": len(selected_pseudo),
        "total_training_records": len(all_records),
        "pseudo_per_class_counts": per_class_counts,
        "conf_threshold": conf_threshold,
        "max_pseudo_per_class": max_pseudo_per_class,
    }
    return all_records, stats


def train_student_trilha_c(
    records: List[Dict],
    val_loader: DataLoader,
    device: torch.device,
    data_dir: str = "data",
    arch: str = "resnet18",
    num_classes: int = 21,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 4e-4,
    backbone_lr_mult: float = 0.25,
    weight_decay: float = 1e-4,
    kd_alpha: float = 0.5,
    checkpoint_path: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]], int, float]:
    """
    Treina o modelo Aluno (`ResNet-18`, <= 12M parâmetros) combinando:
      - Cross-Entropy com os rótulos (hard labels verdadeiros + pseudo-rótulos filtrados)
      - Destilação de Conhecimento (KL-Divergence suave contra `soft_probs` do Professor)
    Seleciona a melhor época ESTRITAMENTE pelo F1 Macro no conjunto `val`.
    """
    seed_everything(seed)
    train_ds = TeacherStudentDataset(
        records=records,
        images_dir=os.path.join(data_dir, "images", "train"),
        transform=get_transforms(img_size=224, is_train=True),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = build_model(arch=arch, num_classes=num_classes, pretrained=True, dropout=0.2, verbose=verbose).to(device)
    eval_criterion = nn.CrossEntropyLoss()

    fc_params = list(model.fc.parameters())
    fc_ids = {id(p) for p in fc_params}
    backbone_params = [p for p in model.parameters() if id(p) not in fc_ids and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_mult},
            {"params": fc_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
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
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        train_preds = []
        train_targets = []

        for images, targets, soft_probs, weights, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            soft_probs = soft_probs.to(device)
            weights = weights.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            logits = model(images)

            # Perda híbrida por amostra: Hard Cross-Entropy + Soft Cross-Entropy (Destilação)
            ce_per_sample = F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.03)
            log_probs = F.log_softmax(logits, dim=1)
            soft_ce_per_sample = -(soft_probs * log_probs).sum(dim=1)

            combined_loss = ((1.0 - kd_alpha) * ce_per_sample + kd_alpha * soft_ce_per_sample) * weights
            loss = combined_loss.mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            preds = torch.argmax(logits.detach(), dim=1)
            train_preds.append(preds.cpu().numpy())
            train_targets.append(targets.cpu().numpy())

        scheduler.step()

        y_tr_true = np.concatenate(train_targets)
        y_tr_pred = np.concatenate(train_preds)
        train_loss = running_loss / max(total_samples, 1)
        train_f1 = float(f1_score(y_tr_true, y_tr_pred, average="macro", labels=list(range(num_classes)), zero_division=0))
        train_acc = float(accuracy_score(y_tr_true, y_tr_pred))

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
                    arch=arch,
                    num_classes=num_classes,
                    epoch=epoch,
                    val_f1=val_f1,
                )

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs or is_best):
            marker = " ★ [Melhor Val F1]" if is_best else ""
            print(
                f"Época {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} (Acc: {val_acc:.3f}){marker}"
            )

    if verbose:
        print(
            f"\nTreinamento do Aluno concluído em {time.time() - t0:.1f}s. "
            f"Melhor época no val: {best_epoch} (Val F1 Macro = {best_val_f1:.4f})."
        )

    model.load_state_dict(best_weights)
    return model, history, best_epoch, best_val_f1


def plot_teacher_noise_diagnostics(
    zs_eval: Dict,
    teacher_cache: Dict,
    imp_teacher: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plota o diagnóstico sistemático do ruído do Professor Zero-Shot vs. Professor Melhorado:
      1) Distribuição de pseudo-rótulos no pool `train` (6.505 imagens), destacando o viés/colapso
         do Zero-Shot bruto (ex: super-predição de `red_fox` e `domestic_cat`) vs. Professor Melhorado.
      2) Comparação de F1 Macro por classe entre Professor Zero-Shot e Professor Melhorado no `test_public`.
    """
    class_names = teacher_cache["class_names"]
    num_classes = len(class_names)

    zs_preds_train = teacher_cache["train"]["logits_trap_prompt"].argmax(dim=1).numpy()
    imp_preds_train = imp_teacher["probs_train"].argmax(axis=1)

    zs_counts = np.bincount(zs_preds_train, minlength=num_classes)
    imp_counts = np.bincount(imp_preds_train, minlength=num_classes)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=130)

    # Subplot 1: Distribuição de pseudo-rótulos gerados no train (6.505 imagens)
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

    # Subplot 2: Top pares de confusão sistemática do Professor Zero-Shot
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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_student_vs_teacher_per_class(
    class_names: List[str],
    f1_piso: np.ndarray,
    f1_teacher_zs: np.ndarray,
    f1_teacher_imp: np.ndarray,
    f1_student: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> pd.DataFrame:
    """
    Constrói a tabela e plota o gráfico comparativo de F1 por classe no `test_public`:
    Piso Supervisionado (0b) vs. Professor Zero-Shot vs. Professor Melhorado vs. Aluno (ResNet-18),
    destacando em quais classes o Aluno supera o Professor.
    """
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

    # Subplot 1: F1 por classe (Professor Zero-Shot vs Professor Melhorado vs Aluno ResNet-18)
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

    # Subplot 2: Ganho do Aluno sobre o Professor Zero-Shot e sobre o Professor Melhorado
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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return df_comp


def build_part1_summary_table(results_list: List[Dict], ceiling_f1: float = 0.6860) -> pd.DataFrame:
    """
    Constrói a tabela comparativa consolidada da Parte 1 (Trilha C: Professor e Aluno),
    incluindo Piso Supervisionado, Professor Zero-Shot, Ablações de Tratamento de Ruído e Teto.
    """
    rows = list(results_list)
    rows.append({
        "Método / Modelo": "Teto de Referência — ResNet-18 (6.505 rótulos verdadeiros)",
        "Papel": "Referência Superior",
        "Pseudo-Rótulos Usados": "-",
        "Val F1 Macro": "-",
        "Test Public F1 Macro": ceiling_f1,
        "Test Public Acurácia": "-",
    })
    return pd.DataFrame(rows)
