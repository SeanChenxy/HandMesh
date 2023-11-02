# ------------------------------------------------------------------------------
# Copyright (c) 2023
# Licensed under the MIT License.
# Written by Xingyu Chen(chenxingyusean@foxmail.com)
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch
from mobrecon.models.modules import Reg2DDecode3D, conv_layer, linear_layer
from mobrecon.models.densestack import DenseStack, DenseStack2, mobile_unit, Reorg
from conv.spiralconv import SpiralConv
from conv.dsconv import DSConv


class Backbone(nn.Module):
    def __init__(self, input_channel=128, uv_channel=21, latent_channel=256):
        super(Backbone, self).__init__()
        self.pre_layer = nn.Sequential(conv_layer(3, input_channel // 2, 3, 2, 1),
                                       mobile_unit(input_channel // 2, input_channel))
        self.thrink = conv_layer(input_channel * 4, input_channel)
        self.dense_stack1 = DenseStack(input_channel, uv_channel)
        self.stack1_remap = conv_layer(uv_channel, uv_channel)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.thrink2 = conv_layer((uv_channel + input_channel), input_channel)
        self.dense_stack2 = DenseStack2(input_channel, uv_channel, final_upsample=False)
        self.mid_proj = conv_layer(1024, latent_channel, 1, 1, 0, bias=False, bn=False, relu=False)
        self.conv = conv_layer(uv_channel, 21, 1, bn=False, relu=False)
        self.uv_reg = nn.Sequential(linear_layer(latent_channel, 128, bn=False), linear_layer(128, 64, bn=False),
                                    linear_layer(64, 2, bn=False, relu=False))
        self.reorg = Reorg()

    def forward(self, x):
        pre_out = self.pre_layer(x)
        pre_out_reorg = self.reorg(pre_out)
        thrink = self.thrink(pre_out_reorg)
        stack1_out = self.dense_stack1(thrink)
        stack1_out_remap = self.stack1_remap(stack1_out)
        input2 = torch.cat((stack1_out_remap, thrink),dim=1)
        thrink2 = self.thrink2(input2)
        stack2_out, stack2_mid = self.dense_stack2(thrink2)
        latent = self.mid_proj(stack2_mid)
        uv_reg = self.uv_reg(self.conv(stack2_out).view(stack2_out.shape[0], 21, -1))

        return latent, uv_reg


class MobRecon(nn.Module):
    """
        https://arxiv.org/pdf/2112.02753.pdf
    """
    def __init__(self, args, spiral_indices, up_transform):
        super(MobRecon, self).__init__()

        self.uv_channel = 21
        self.input_channel = 128
        self.up_transform = up_transform
        self.backbone = Backbone(self.input_channel, 24, args.out_channels[-1])
        self.decoder3d = Reg2DDecode3D(self.input_channel * 2, args.out_channels, spiral_indices, up_transform, self.uv_channel, meshconv=(SpiralConv, DSConv)[args.dsconv])

    def forward(self, x):
        latent, pred2d_pt = self.backbone(x)
        pred3d = self.decoder3d(pred2d_pt, latent)

        return {'mesh_pred': pred3d,
                'uv_pred': pred2d_pt
                }
