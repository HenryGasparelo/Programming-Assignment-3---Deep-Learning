import torch
import torch.nn as nn
from torchvision import models
from src.utils import check_parameter_limit


class SmallCNN(nn.Module):
    """
    Parte 0 (a): CNN pequena treinada do zero para estabelecer a linha de base sem pré-treino.
    Possui 4 blocos convolucionais com BatchNorm, ReLU, MaxPool, seguidos de AdaptiveAvgPool e classificador.
    (~425k parâmetros, bem abaixo do limite de 12M).
    """

    def __init__(self, num_classes: int = 21, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Bloco 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Pool global -> 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(
    arch: str = "resnet18",
    num_classes: int = 21,
    pretrained: bool = True,
    dropout: float = 0.2,
    verbose: bool = True,
) -> nn.Module:
    """
    Constrói o modelo especificado e verifica automaticamente o limite de 12M parâmetros.
    Opções principais:
      - 'small_cnn': CNN pequena treinada do zero (Parte 0a)
      - 'resnet18': ResNet-18 (Parte 0b e backbone principal do PA, ~11.19M parâmetros)
      - 'mobilenet_v3_small': MobileNetV3-Small (~1.5M parâmetros)
      - 'efficientnet_b0': EfficientNet-B0 (~4.0M parâmetros)
    """
    arch = arch.lower()
    if arch == "small_cnn":
        model = SmallCNN(num_classes=num_classes, dropout=dropout)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Arquitetura não suportada: {arch}")

    assert check_parameter_limit(model, max_params=12_000_000, verbose=verbose), (
        f"Modelo {arch} excedeu o limite de 12 milhões de parâmetros!"
    )
    return model
