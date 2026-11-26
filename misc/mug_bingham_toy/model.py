"""Full, randomly initialized VGG16 with a nine-parameter Bingham head.

This self-contained definition follows the standard VGG16 configuration D:
13 convolutions, five max pools, adaptive 7x7 pooling, and the complete
25088 -> 4096 -> 4096 -> 9 classifier. Its module/state-dict names match the
torchvision VGG16 layout. No pretrained weights or frozen parameters are used.
The only initialization change is a small final linear layer, so training
starts near a uniform Bingham law without exactly repeated eigenvalues.
"""
from __future__ import annotations

import torch
from torch import nn


class VGG16Bingham(nn.Module):
    """Standard VGG16 accepting whole RGB images and returning float32 [B,9].

    All trainable tensors remain float32. With ``amp_bfloat16=True`` CUDA
    convolution/classifier features use autocast, while the final linear layer
    runs in float32. The Bingham loss performs its own float64 arithmetic.
    Input normalization belongs to the data pipeline, not this network.
    """

    def __init__(self, amp_bfloat16: bool = False):
        super().__init__()
        configuration = (64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                         512, 512, 512, "M", 512, 512, 512, "M")
        layers = []
        channels = 3
        for value in configuration:
            if value == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend((nn.Conv2d(channels, value, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)))
                channels = value
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(p=0.5),
            nn.Linear(4096, 9))
        self.amp_bfloat16 = amp_bfloat16
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.classifier[-1].weight, 0, 1e-4)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("VGG16 input must have shape [batch,3,height,width]")
        if min(images.shape[-2:]) < 32:
            raise ValueError("VGG16 needs both spatial dimensions at least 32")
        use_amp = self.amp_bfloat16 and images.device.type == "cuda"
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                            enabled=use_amp):
            features = self.avgpool(self.features(images))
            features = torch.flatten(features, 1)
            for layer in self.classifier[:-1]:
                features = layer(features)
        with torch.autocast(device_type=images.device.type, enabled=False):
            return self.classifier[-1](features.float())


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Counts recorded in provenance to make full-network training auditable."""
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters()
                         if parameter.requires_grad),
        "frozen": sum(parameter.numel() for parameter in model.parameters()
                      if not parameter.requires_grad),
    }
