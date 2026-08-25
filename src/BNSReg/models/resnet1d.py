# SPDX-License-Identifier: MIT
#
# Ported from ml4gw.nn.resnet.resnet_1d (https://github.com/ML4GW/ml4gw),
# itself derived from torchvision's ResNet
# (https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).
#
# Modifications made on 2026-08-24:
#   - Copied verbatim to avoid an ml4gw runtime dependency for this one class.
#   - Default norm_layer changed from ml4gw's GroupNorm1DGetter to
#     nn.BatchNorm1d (avoids porting ml4gw.nn.norm as well).

from typing import Callable, List, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor


def convN(in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1) -> nn.Conv1d:
    if not kernel_size % 2:
        raise ValueError("Can't use even sized kernels")
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=dilation * int(kernel_size // 2), groups=groups, bias=False, dilation=dilation)


def conv1(in_planes, out_planes, stride=1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = convN(inplanes, planes, kernel_size, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convN(planes, planes, kernel_size)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """1D ResNet: conv1(k=7,s=2) -> bn -> relu -> maxpool -> residual layers
    -> AdaptiveAvgPool1d(1) -> fc. See ml4gw.nn.resnet.resnet_1d for full docs.
    """
    block = BasicBlock

    def __init__(
        self,
        in_channels: int,
        layers: List[int],
        classes: int,
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[List[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self._norm_layer = norm_layer or nn.BatchNorm1d

        if stride_type is None:
            stride_type = ["stride"] * (len(layers) - 1)
        if len(stride_type) != (len(layers) - 1):
            raise ValueError(f"'stride_type' should be None or a {len(layers) - 1}-element tuple, got {stride_type}")

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        residual_layers = [self._make_layer(64, layers[0], kernel_size)]
        for i, (num_blocks, stride) in enumerate(zip(layers[1:], stride_type)):
            block_size = 64 * 2 ** (i + 1)
            residual_layers.append(self._make_layer(block_size, num_blocks, kernel_size, stride=2, stride_type=stride))
        self.residual_layers = nn.ModuleList(residual_layers)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(block_size * self.block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes, blocks, kernel_size=3, stride=1, stride_type="stride") -> nn.Sequential:
        block, norm_layer = self.block, self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride_type == "dilation":
            self.dilation *= stride
            stride = 1
        elif stride_type != "stride":
            raise ValueError(f"Unknown stride type {stride_type}")

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, kernel_size, stride, downsample,
                         self.groups, self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size, groups=self.groups,
                                 base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in self.residual_layers:
            x = layer(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)
