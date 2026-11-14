import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from src.data import load_class_mapping, CameraTrapDataset


def format_species_prompts(class_names: List[str], template: str = "a camera trap photo of a {}") -> List[str]:
    """Formata os nomes das classes do ENA24 em prompts textuais para o BioCLIP."""
    return [template.format(cname.replace("_", " ")) for cname in class_names]


@torch.no_grad()
def extract_teacher_features_and_zero_shot(
    model_id: str = "hf-hub:imageomics/bioclip",
    data_dir: str = "data",
    cache_dir: str = "checkpoints",
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    force_recompute: bool = False,
) -> Dict:
    """
    Carrega o Foundation Model aberto (BioCLIP 'imageomics/bioclip'),
    extrai embeddings visuais e logits zero-shot para `train` (6.505 imagens) e
    `test_public` (811 imagens) — NUNCA olhando o `val` conforme regra estrita da Trilha C.
    Salva os resultados em cache binário para reuso instantâneo e reprodutibilidade.
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
    Avalia o desempenho Zero-Shot do Professor (BioCLIP) no `test_public`:
    - Prompt com nomes literais ('american black bear')
    - Prompt contextualizado ('a camera trap photo of a american black bear')
    - Diagnóstico de acurácia em imagens diurnas (coloridas) vs. noturnas (infravermelho)
    - Identificação dos pares sistemáticos da matriz de ruído
    """
    class_names = teacher_cache["class_names"]
    num_classes = len(class_names)
    y_true = teacher_cache["test_public"]["y_true"]

    # 1. Prompt com nomes puros
    logits_raw = teacher_cache["test_public"]["logits_raw_names"]
    preds_raw = logits_raw.argmax(dim=1).numpy()
    f1_raw = float(f1_score(y_true, preds_raw, average="macro", labels=list(range(num_classes)), zero_division=0))
    acc_raw = float(accuracy_score(y_true, preds_raw))

    # 2. Prompt contextualizado
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
    Constrói o Professor Melhorado com os rótulos humanos do orçamento fornecido,
    sem JAMAIS tocar no conjunto `val`.
    Combina:
      1) Prior Zero-Shot desenviesado por alinhamento marginal no pool `train`
      2) Protótipos visuais de cosseno das amostras rotuladas no espaço BioCLIP
      3) Adaptador linear L2 ajustado exclusivamente nas amostras rotuladas
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

    zs_probs_train = torch.softmax(teacher_cache["train"]["logits_trap_prompt"] / zs_temp, dim=1).numpy()
    zs_probs_test = torch.softmax(teacher_cache["test_public"]["logits_trap_prompt"] / zs_temp, dim=1).numpy()

    marginal_prior = np.clip(zs_probs_train.mean(axis=0, keepdims=True), 1e-6, 1.0)
    zs_deb_train = zs_probs_train / (marginal_prior ** 0.5)
    zs_deb_train /= zs_deb_train.sum(axis=1, keepdims=True)

    zs_deb_test = zs_probs_test / (marginal_prior ** 0.5)
    zs_deb_test /= zs_deb_test.sum(axis=1, keepdims=True)

    prototypes = np.zeros((num_classes, X_train.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        mask = labeled_targets == c
        if mask.any():
            p = X_labeled[mask].mean(axis=0)
            prototypes[c] = p / max(np.linalg.norm(p), 1e-8)

    proto_probs_train = torch.softmax(torch.tensor(X_train @ prototypes.T) * 16.0, dim=1).numpy()
    proto_probs_test = torch.softmax(torch.tensor(X_test @ prototypes.T) * 16.0, dim=1).numpy()

    clf = LogisticRegression(C=c_reg, max_iter=1000, random_state=seed)
    clf.fit(X_labeled, labeled_targets)
    lr_probs_train = clf.predict_proba(X_train)
    lr_probs_test = clf.predict_proba(X_test)

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
    Trata o ruído dos pseudo-rótulos para o treinamento do Aluno:
      - Filtra predições com confiança máxima < conf_threshold
      - Limita o número de pseudo-rótulos por classe (cota balanceada anti-viés)
      - Suaviza a distribuição com temperatura T para destilação suave
      - Ancola o treinamento com repetições dos rótulos humanos verdadeiros
    """
    train_ids = teacher_cache["train"]["image_ids"]
    num_classes = probs_train.shape[1]

    human_records = []
    labeled_set = set(labeled_id_to_target.keys())
    for img_id, target in labeled_id_to_target.items():
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[target] = 1.0
        for _ in range(human_oversample):
            human_records.append({
                "image_id": img_id,
                "label": target,
                "soft_probs": one_hot,
                "weight": 1.0,
                "is_human": True,
            })

    candidates_by_class = {c: [] for c in range(num_classes)}
    for i, img_id in enumerate(train_ids):
        if str(img_id) in labeled_set:
            continue
        p = probs_train[i]
        top_cls = int(p.argmax())
        top_conf = float(p[top_cls])

        if top_conf >= conf_threshold:
            log_p = np.log(np.clip(p, 1e-8, 1.0)) / temperature
            soft_p = np.exp(log_p - log_p.max())
            soft_p /= soft_p.sum()
            candidates_by_class[top_cls].append({
                "image_id": str(img_id),
                "label": top_cls,
                "soft_probs": soft_p.astype(np.float32),
                "confidence": top_conf,
                "weight": float(np.clip(top_conf, 0.3, 1.0)),
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
