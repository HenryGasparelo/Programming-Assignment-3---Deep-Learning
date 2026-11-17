#!/usr/bin/env python3
"""
Script oficial de inferência para o PA3 (ENA24-detection).

Uso:
    python predict.py --imagens PASTA --saida preds.csv

Requisitos atendidos (Seção 4 do enunciado):
- Roda estritamente em CPU e sem acesso à internet (pretrained=False).
- Imprime o número exato de parâmetros do modelo: sum(p.numel() for p in model.parameters()).
- Respeita o limite de <= 12.000.000 parâmetros (ResNet-18 Aluno: 11.187.285 parâmetros).
- Gera arquivo CSV com colunas `image_id,label` usando os nomes oficiais de `data/classes.csv`.
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data import get_transforms, load_class_mapping
from src.models import build_model
from src.utils import check_parameter_limit, count_parameters

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FALLBACK_CHECKPOINTS = [
    "checkpoints/best_model.pt",
    "checkpoints/part2_student_trilha_c_b20_s42.pt",
    "checkpoints/part2_student_trilha_c_b5_s42.pt",
    "checkpoints/part1_student_c3_improved_teacher_b5_s42.pt",
    "checkpoints/part1_student_c2_filtered_zs_s42.pt",
    "checkpoints/part0b_resnet18_b20_s42.pt",
    "checkpoints/part0b_resnet18_b5_s42.pt",
]


class InferenceFolderDataset(Dataset):
    """Dataset leve para inferência em uma pasta de imagens arbitrária."""

    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        image_id = img_path.stem
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image_id, image


def resolve_checkpoint_path(requested_path: str, project_root: Path) -> Path:
    """Localiza o checkpoint solicitado ou garante a criação de checkpoints/best_model.pt."""
    req = Path(requested_path)
    if not req.is_absolute():
        req = project_root / req

    if req.exists():
        return req

    for candidate_rel in FALLBACK_CHECKPOINTS:
        candidate = project_root / candidate_rel
        if candidate.exists():
            if req.name == "best_model.pt":
                req.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, req)
                return req
            return candidate

    raise FileNotFoundError(
        f"Nenhum checkpoint encontrado em '{requested_path}' nem nos caminhos de fallback: "
        f"{FALLBACK_CHECKPOINTS}"
    )


def collect_images(folder: Path) -> List[Path]:
    """Coleta todas as imagens válidas da pasta informada (ordenadas por nome)."""
    if not folder.exists():
        raise FileNotFoundError(f"A pasta de imagens não existe: {folder}")

    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    if not images:
        images = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
    images.sort(key=lambda p: p.name)
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferência em CPU para classificação de fauna em câmeras trap (ENA24-detection)."
    )
    parser.add_argument(
        "--imagens",
        "-imagens",
        type=str,
        required=True,
        help="Caminho para a pasta contendo as imagens de entrada.",
    )
    parser.add_argument(
        "--saida",
        "-saida",
        type=str,
        required=True,
        help="Caminho do arquivo CSV de saída (colunas: image_id,label).",
    )
    parser.add_argument(
        "--pesos",
        "-pesos",
        type=str,
        default="checkpoints/best_model.pt",
        help="Caminho para os pesos do modelo (padrão: checkpoints/best_model.pt).",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="data/classes.csv",
        help="Caminho para o arquivo de mapeamento de classes (padrão: data/classes.csv).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do lote para inferência em CPU (padrão: 32).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    # 1. Dispositivo estritamente em CPU conforme especificação da Seção 4
    device = torch.device("cpu")

    # 2. Carrega mapeamento oficial de classes (data/classes.csv)
    classes_path = Path(args.classes)
    if not classes_path.is_absolute():
        classes_path = project_root / classes_path
    _, idx_to_label, _ = load_class_mapping(str(classes_path))
    num_classes = len(idx_to_label)

    # 3. Instancia o modelo Aluno (ResNet-18) offline (pretrained=False) e carrega pesos em CPU
    ckpt_path = resolve_checkpoint_path(args.pesos, project_root)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "resnet18") if isinstance(checkpoint, dict) else "resnet18"

    model = build_model(
        arch=arch,
        num_classes=num_classes,
        pretrained=False,
        dropout=0.2,
        verbose=False,
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Imprime o número exato de parâmetros e valida o teto de 12M
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo carregado: {arch} ({ckpt_path.name})")
    print(f"Número de parâmetros do modelo: {num_params} ({num_params:,})")
    check_parameter_limit(model, max_params=12_000_000)

    # 5. Prepara o DataLoader de inferência
    images_dir = Path(args.imagens)
    image_paths = collect_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"Nenhuma imagem encontrada na pasta: {images_dir}")

    transform = get_transforms(img_size=224, is_train=False)
    dataset = InferenceFolderDataset(image_paths=image_paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 6. Executa a inferência em CPU sem cálculo de gradiente
    results = []
    with torch.no_grad():
        for image_ids, images in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            for img_id, pred_idx in zip(image_ids, preds):
                results.append({
                    "image_id": str(img_id),
                    "label": idx_to_label[int(pred_idx)],
                })

    # 7. Salva o CSV no formato exigido: image_id,label
    output_path = Path(args.saida)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results, columns=["image_id", "label"])
    df_out.to_csv(output_path, index=False)
    print(f"Predições salvas com sucesso em '{output_path}' ({len(df_out)} imagens processadas).")


if __name__ == "__main__":
    main()
