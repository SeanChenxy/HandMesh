# ------------------------------------------------------------------------------
# Copyright (c) 2021
# Licensed under the MIT License.
# Written by Xingyu Chen(chenxingyusean@foxmail.com)
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch
import torch.nn.functional as F
from .network import ConvBlock, SpiralConv, Pool, ParallelDeblock, SelfAttention
from .resnet import resnet18, resnet50
from .loss import l1_loss, bce_loss, normal_loss, edge_length_loss

class EncodeStage1(nn.Module):
    def __init__(self, backbone):
        super(EncodeStage1, self).__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x4, x3, x2, x1


class EncodeStage2(nn.Module):
    def __init__(self, backbone, in_channel):
        super(EncodeStage2, self).__init__()
        self.reduce = nn.Sequential(ConvBlock(in_channel, in_channel, relu=True, norm='bn'),
                                    ConvBlock(in_channel, 128, relu=True, norm='bn'),
                                    ConvBlock(128, 64, kernel_size=1, padding=0, relu=False, norm='bn'))
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, x):
        x0 = self.reduce(x)
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x4, x3, x2, x1


class EncodeStage3(nn.Module):
    def __init__(self, backbone, in_channel):
        super(EncodeStage3, self).__init__()
        self.reduce = nn.Sequential(ConvBlock(in_channel, in_channel, relu=True, norm='bn'),
                                    ConvBlock(in_channel, 128, relu=True, norm='bn'),
                                    ConvBlock(128, 64, kernel_size=1, padding=0, relu=False, norm='bn'))
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, x):
        x = self.reduce(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, x4, x3, x2, x1


class CMR_SG(nn.Module):
    """
    Implementation of CMR_SG.
    :param spiral_indices: pre-defined spiral sample
    :param up_transform: pre-defined upsample matrix
    :param relation: This implementation only adopts tip-based aggregation.
                     You can employ more sub-poses by enlarge relation list.
    """
    def __init__(self, args, spiral_indices, up_transform):
        super(CMR_SG, self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u.size(0) for u in self.up_transform] + [self.up_transform[-1].size(1)]
        self.uv_channel = 21
        self.relation = [[4, 8], [4, 12], [4, 16], [4, 20], [8, 12], [8, 16], [8, 20], [12, 16], [12, 20], [16, 20]]

        backbone, self.latent_size = self.get_backbone(args.backbone)
        self.backbone1 = EncodeStage1(backbone)

        backbone2, _ = self.get_backbone(args.backbone)
        self.backbone2 = EncodeStage2(backbone2, 64 + 1)

        backbone3, _ = self.get_backbone(args.backbone)
        self.backbone3 = EncodeStage3(backbone3, 64+self.uv_channel+len(self.relation))

        self.mask_delayer = nn.ModuleList([ConvBlock(self.latent_size[2] + self.latent_size[1], self.latent_size[2], kernel_size=3, relu=True, norm='bn'),
                                          ConvBlock(self.latent_size[3] + self.latent_size[2], self.latent_size[3], kernel_size=3, relu=True, norm='bn'),
                                          ConvBlock(self.latent_size[4] + self.latent_size[3], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                          ConvBlock(self.latent_size[4], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                          ])
        self.mask_head = ConvBlock(self.latent_size[4], 1, kernel_size=3, padding=1, relu=False, norm=None)

        self.uv_delayer = nn.ModuleList([ConvBlock(self.latent_size[2] + self.latent_size[1], self.latent_size[2], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[3] + self.latent_size[2], self.latent_size[3], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[4] + self.latent_size[3], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[4], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                         ])
        self.uv_head = ConvBlock(self.latent_size[4], self.uv_channel, kernel_size=3, padding=1, relu=False, norm=None)

        self.uvm_delayer = nn.ModuleList([ConvBlock(self.latent_size[2] + self.latent_size[1], self.latent_size[2], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[3] + self.latent_size[2], self.latent_size[3], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[4] + self.latent_size[3], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[4], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                         ])
        self.uvm_head = ConvBlock(self.latent_size[4], self.uv_channel+1, kernel_size=3, padding=1, relu=False, norm=None)

        # 3D decoding
        self.attention = SelfAttention(self.latent_size[0])
        self.de_layers = nn.ModuleList()
        self.de_layers.append(nn.Linear(self.latent_size[0], self.num_vert[-1] * self.out_channels[-1]))
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layers.append(ParallelDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1],
                                                      self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(ParallelDeblock(self.out_channels[-idx] + 3, self.out_channels[-idx - 1],
                                                      self.spiral_indices[-idx - 1]))
        self.heads = nn.ModuleList()
        for oc, sp_idx in zip(self.out_channels[::-1], self.spiral_indices[::-1]):
            self.heads.append(SpiralConv(oc, self.in_channels, sp_idx))

    def get_backbone(self, backbone, pretrained=True):
        if '50' in backbone:
            basenet = resnet50(pretrained=pretrained)
            latent_channel = (1000, 2048, 1024, 512, 256)
        elif '18' in backbone:
            basenet = resnet18(pretrained=pretrained)
            latent_channel = (1000, 512, 256, 128, 64)
        else:
            raise Exception("Not supported", backbone)

        return basenet, latent_channel

    def decoder(self, x):
        x = self.attention(x)
        num_layers = len(self.de_layers)
        num_features = num_layers - 1
        hierachy_pred = []
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert[-1], self.out_channels[-1])
            else:
                x = layer(x, self.up_transform[num_features - i])
                pred = self.heads[i - 1](x)
                if i > 1:
                    pred = (pred + Pool(hierachy_pred[-1], self.up_transform[num_features-i]))/2
                hierachy_pred.append(pred)
                x = torch.cat((x, pred), 2)

        return hierachy_pred[::-1]

    def mask_decoder(self, z):
        x = z[0]
        for i, de in enumerate(self.mask_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < 3:
                x = torch.cat((x, z[i+1]), dim=1)
            x = de(x)
        pred = torch.sigmoid(self.mask_head(x))

        return pred

    def uv_decoder(self, z):
        x = z[0]
        for i, de in enumerate(self.uv_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < 3:
                x = torch.cat((x, z[i+1]), dim=1)
            x = de(x)
        pred = torch.sigmoid(self.uv_head(x))

        return pred

    def uvm_decoder(self, z):
        x = z[0]
        for i, de in enumerate(self.uvm_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < 3:
                x = torch.cat((x, z[i + 1]), dim=1)
            x = de(x)
        pred = torch.sigmoid(self.uvm_head(x))

        return pred

    def forward(self, x):
        z1 = self.backbone1(x)
        pred1 = self.mask_decoder(z1[1:])
        z2 = self.backbone2(torch.cat([z1[0], pred1], 1))
        pred2 = self.uv_decoder(z2[1:])
        z3 = self.backbone3(torch.cat([z2[0], pred2] + [pred2[:, i].sum(dim=1, keepdim=True) for i in self.relation], 1))
        pred3 = self.decoder(z3[0])
        pred4 = self.uvm_decoder(z3[1:])

        return {'mesh_pred': pred3,
                'uv_pred': pred4[:, :self.uv_channel],
                'mask_pred': pred4[:, self.uv_channel],
                'mask_prior': pred1[:, 0],
                'uv_prior': pred2,
                }

    def loss(self, **kwargs):
        loss_dict = dict()
        loss = 0.
        for i in range(len(kwargs['gt'])):
            loss += l1_loss(kwargs['pred'][i], kwargs['gt'][i])
            if i == 0:
                loss_dict['l1_loss'] = loss.clone()
        loss_dict['uv_loss'] = 10 * bce_loss(kwargs['uv_pred'], kwargs['uv_gt'])
        loss_dict['uv_prior_loss'] = 10 * bce_loss(kwargs['uv_prior'], kwargs['uv_gt'])
        loss_dict['mask_loss'] = 0.5 * bce_loss(kwargs['mask_pred'], kwargs['mask_gt'])
        loss_dict['mask_prior_loss'] = 0.5 * bce_loss(kwargs['mask_prior'], kwargs['mask_gt'])
        loss_dict['normal_loss'] = 0.1 * normal_loss(kwargs['pred'][0], kwargs['gt'][0], kwargs['face'])
        loss_dict['edge_loss'] = edge_length_loss(kwargs['pred'][0], kwargs['gt'][0], kwargs['face'])
        loss += loss_dict['uv_loss'] + loss_dict['normal_loss'] + loss_dict['edge_loss'] + loss_dict['uv_prior_loss'] + loss_dict['mask_loss'] + loss_dict['mask_prior_loss']
        loss_dict['loss'] = loss

        return loss_dict
