import os
from typing import Dict, Tuple, Optional, List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_class_mapping(classes_csv: str = "data/classes.csv") -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """Carrega o mapeamento oficial de classes (label <-> idx)."""
    df = pd.read_csv(classes_csv).sort_values("idx").reset_index(drop=True)
    class_to_idx = {row["label"]: int(row["idx"]) for _, row in df.iterrows()}
    idx_to_class = {int(row["idx"]): row["label"] for _, row in df.iterrows()}
    class_names = [idx_to_class[i] for i in range(len(df))]
    return class_to_idx, idx_to_class, class_names


def get_transforms(img_size: int = 224, is_train: bool = True) -> T.Compose:
    """
    Retorna as transformações de pré-processamento e aumentação de dados.
    Nota: Redimensionamento direto preserva as bordas do quadro, evitando cortar animais pequenos nos cantos.
    """
    if is_train:
        return T.Compose([
            T.RandomResizedCrop((img_size, img_size), scale=(0.6, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class CameraTrapDataset(Dataset):
    """Dataset para imagens de câmeras trap do ENA24-detection."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[T.Compose] = None,
        return_id: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_id = return_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        img_name = image_id if image_id.endswith(".jpg") else f"{image_id}.jpg"
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if "label" in row and self.class_to_idx is not None and pd.notna(row["label"]):
            label = self.class_to_idx[row["label"]]
            if self.return_id:
                return image, label, image_id
            return image, label

        if self.return_id:
            return image, image_id
        return image


def get_supervised_dataloaders(
    data_dir: str = "data",
    budget: int = 5,
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str], List[str]]:
    """
    Cria os DataLoaders supervisionados para treino (rotulos_1/5/20), validação (val) e teste público (test_public).
    """
    classes_csv = os.path.join(data_dir, "classes.csv")
    class_to_idx, idx_to_class, class_names = load_class_mapping(classes_csv)

    train_csv = os.path.join(data_dir, "labels", f"rotulos_{budget}.csv")
    val_csv = os.path.join(data_dir, "labels", "val.csv")
    test_csv = os.path.join(data_dir, "labels", "test_public.csv")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    train_ds = CameraTrapDataset(
        df=df_train,
        images_dir=os.path.join(data_dir, "images", "train"),
        class_to_idx=class_to_idx,
        transform=get_transforms(img_size=img_size, is_train=True),
    )
    val_ds = CameraTrapDataset(
        df=df_val,
        images_dir=os.path.join(data_dir, "images", "val"),
        class_to_idx=class_to_idx,
        transform=get_transforms(img_size=img_size, is_train=False),
    )
    test_ds = CameraTrapDataset(
        df=df_test,
        images_dir=os.path.join(data_dir, "images", "test_public"),
        class_to_idx=class_to_idx,
        transform=get_transforms(img_size=img_size, is_train=False),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class, class_names
