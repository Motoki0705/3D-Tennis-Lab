"""
HRNet with 3D stem.

This module replaces the first two 2D conv layers (conv1, conv2) with 3D convs
that operate on temporal stacks (frames_in). The temporal axis is then
aggregated (mean/center/max) to produce a 2D feature map that flows into the
standard HRNet stages (layer1, stage2..4, deconv, final_layers).

Weight compatibility:
- The stem keeps the same parameter names: conv1, bn1, conv2, bn2.
- However, conv1/conv2 are nn.Conv3d and bn1/bn2 are nn.BatchNorm3d.
- Use the provided inflate tool to convert 2D weights to 3D shapes when loading
  a checkpoint trained with the 2D stem.
"""

from __future__ import absolute_import, division, print_function

import logging
import os
from typing import List

import torch
import torch.nn as nn

from .base_hrnet import (
    HighResolutionModule,
    BN_MOMENTUM,
    blocks_dict,
)


logger = logging.getLogger(__name__)


class HRNet3DStem(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(HRNet3DStem, self).__init__()

        # Common
        self._frames_in: int = cfg["frames_in"]
        self._frames_out: int = cfg["frames_out"]
        self._out_scales: List[int] = cfg["out_scales"]

        # 2D stem params (spatial)
        self._stem_strides = cfg["MODEL"]["EXTRA"]["STEM"]["STRIDES"]  # [s1, s2]
        self._stem_inplanes = cfg["MODEL"]["EXTRA"]["STEM"]["INPLANES"]

        # 3D stem params (temporal)
        stem3d = cfg["MODEL"]["EXTRA"].get("STEM3D", {})
        # depth kernel sizes for conv1/conv2 in time axis (kD1, kD2)
        kds = stem3d.get("KERNELS_D", None)
        if isinstance(kds, (list, tuple)) and len(kds) == 2:
            self._stem3d_kd1, self._stem3d_kd2 = int(kds[0]), int(kds[1])
        else:
            kd = int(stem3d.get("KERNEL_D", 3))
            self._stem3d_kd1 = kd
            self._stem3d_kd2 = kd
        # depth strides for conv1/conv2 in time axis (sD1, sD2)
        sds = stem3d.get("STRIDES_D", None)
        if isinstance(sds, (list, tuple)) and len(sds) == 2:
            self._stem3d_sd1, self._stem3d_sd2 = int(sds[0]), int(sds[1])
        else:
            self._stem3d_sd1 = 1
            self._stem3d_sd2 = 1

        self._agg: str = str(stem3d.get("AGG", "mean")).lower()  # 'mean'|'max'|'center'

        # 3D stem
        # Input expected: (B, 3*frames_in, H, W) -> reshape to (B, 3, F, H, W)
        self.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=self._stem_inplanes,
            kernel_size=(self._stem3d_kd1, 3, 3),
            stride=(self._stem3d_sd1, self._stem_strides[0], self._stem_strides[0]),
            padding=(self._stem3d_kd1 // 2, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self._stem_inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            in_channels=self._stem_inplanes,
            out_channels=self._stem_inplanes,
            kernel_size=(self._stem3d_kd2, 3, 3),
            stride=(self._stem3d_sd2, self._stem_strides[1], self._stem_strides[1]),
            padding=(self._stem3d_kd2 // 2, 1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(self._stem_inplanes, momentum=BN_MOMENTUM)

        # Stages, same as base HRNet (2D)
        self.stage1_cfg = cfg["MODEL"]["EXTRA"]["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, self._stem_inplanes, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg["MODEL"]["EXTRA"]["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg["MODEL"]["EXTRA"]["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg["MODEL"]["EXTRA"]["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        self.num_deconvs = cfg["MODEL"]["EXTRA"]["DECONV"]["NUM_DECONVS"]
        self.deconv_config = cfg["MODEL"]["EXTRA"]["DECONV"]
        self.pretrained_layers = cfg["MODEL"]["EXTRA"]["PRETRAINED_LAYERS"]

        self.deconv_layers = self._make_deconv_layers(cfg, pre_stage_channels)
        self.final_layers = self._make_final_layers(cfg, pre_stage_channels)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_final_layers(self, cfg, input_channels):
        out_channels_per_scale = cfg["MODEL"]["EXTRA"]["OUT_CHANNELS_PER_SCALE"]
        final_layers = []
        for i in range(len(self._out_scales)):
            scale = self._out_scales[i]
            final_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels[scale],
                        out_channels=out_channels_per_scale[scale],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                )
            )
        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, cfg, pre_stage_channels):
        if cfg["MODEL"]["EXTRA"]["DECONV"]["NUM_DECONVS"] == 0:
            return None

        num_deconvs = cfg["MODEL"]["EXTRA"]["DECONV"]["NUM_DECONVS"]
        kernels = cfg["MODEL"]["EXTRA"]["DECONV"]["KERNEL_SIZE"]
        assert len(kernels) == num_deconvs, "KERNEL_SIZE length must match NUM_DECONVS"

        # Build per-deconv, per-scale upsamplers
        deconv_layers = []
        for i in range(num_deconvs):
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(kernels[i])
            per_scale = []
            for scale in self._out_scales:
                in_ch = pre_stage_channels[scale]
                out_ch = in_ch
                per_scale.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=deconv_kernel,
                            stride=2,
                            padding=padding,
                            output_padding=output_padding,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                )
            deconv_layers.append(nn.ModuleList(per_scale))

        return nn.ModuleList(deconv_layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _aggregate_temporal(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) -> (B, C, H, W)
        if self._agg == "mean":
            return x.mean(dim=2)
        elif self._agg == "max":
            return x.max(dim=2).values
        elif self._agg == "center":
            center = x.shape[2] // 2
            return x[:, :, center, :, :]
        else:
            raise ValueError(f"Unknown aggregation method: {self._agg}")

    def forward(self, x):
        # Input x: (B, 3*F, H, W)
        B, C, H, W = x.shape
        F = self._frames_in
        assert C == 3 * F, f"Input channels ({C}) != 3*frames_in ({3 * F})"
        x = x.view(B, 3, F, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Aggregate temporal dimension -> 2D feature map
        x = self._aggregate_temporal(x)

        # Standard HRNet 2D pipeline from here
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        y_out = {}
        for scale in self._out_scales:
            x = y_list[scale]
            for i in range(self.num_deconvs):
                x = self.deconv_layers[i][scale](x)
            y = self.final_layers[scale](x)
            y_out[scale] = y
        return y_out

    def init_weights(self, pretrained=""):
        logger.info("=> init weights from normal distribution (3D stem)")
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location="cpu")
            logger.info("=> loading pretrained model {}".format(pretrained))
            model_dict = self.state_dict()
            if isinstance(pretrained_dict, dict) and "state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["state_dict"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info("=> loading {} pretrained model {}".format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
