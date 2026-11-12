"""ResNet-18 de un canal y dos clases para Dike."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _conv_pair(value: tuple[int, ...] | str) -> tuple[int, int]:
    """Devuelve el alto y el ancho de un parámetro de convolución 2D."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("la convolución de ResNet-18 debe ser de dos dimensiones")
    return value[0], value[1]


def device() -> torch.device:
    """Devuelve la GPU si está disponible y, si no, la CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    pretrained: bool = True, seed: int | None = None
) -> nn.Module:
    """ResNet-18 de un canal y exactamente dos logits.

    Si se usan pesos ImageNet, la primera convolución RGB->gris se obtiene
    sumando los tres kernels de entrada. Sumar preserva la respuesta esperada
    de un canal gris replicado; promediar reduciría la activación por 3.
    """
    if seed is not None:
        torch.manual_seed(seed)

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    old_conv = model.conv1
    gray_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=_conv_pair(old_conv.kernel_size),
        stride=_conv_pair(old_conv.stride),
        padding=_conv_pair(old_conv.padding),
        bias=old_conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            gray_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
            if old_conv.bias is not None and gray_conv.bias is not None:
                gray_conv.bias.copy_(old_conv.bias)
    model.conv1 = gray_conv
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def freeze_trained_weights(model: nn.Module, frozen: bool) -> None:
    """Impide que cambien los pesos ya entrenados cuando ``frozen`` es verdadero.

    La capa ``fc`` sigue pudiendo cambiar.
    """
    for name, param in model.named_parameters():
        param.requires_grad = (not frozen) or name.startswith("fc.")
