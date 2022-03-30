# ------------------------------------------------------------------------------
# Copyright (c) 2021
# Licensed under the MIT License.
# Written by Xingyu Chen(chenxingyusean@foxmail.com)
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch
import torch.nn.functional as F
from cmr.network import SpiralConv, Pool
from cmr.resnet import resnet18, resnet50


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.relu(self.conv(out))
        return out


class Encoder(nn.Module):
    def __init__(self, backbone):
        super(Encoder, self).__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class YTBHand(nn.Module):
    """
    re-implementation of YoutubeHand.
    See https://openaccess.thecvf.com/content_CVPR_2020/papers/Kulon_Weakly-Supervised_Mesh-Convolutional_Hand_Reconstruction_in_the_Wild_CVPR_2020_paper.pdf
    """
    def __init__(self, args, spiral_indices, up_transform):
        super(YTBHand, self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u.size(0) for u in self.up_transform] + [self.up_transform[-1].size(1)]
        self.uv_channel = 21

        backbone, self.latent_size = self.get_backbone(args.backbone)
        self.backbone = Encoder(backbone)

        # 3D decoding
        self.de_layers = nn.ModuleList()
        self.de_layers.append(nn.Linear(self.latent_size[0], self.num_vert[-1] * self.out_channels[-1]))
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layers.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1],
                                                      self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1],
                                                      self.spiral_indices[-idx - 1]))
        self.heads = SpiralConv(self.out_channels[0], self.in_channels, self.spiral_indices[0])

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
        num_layers = len(self.de_layers)
        num_features = num_layers - 1
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert[-1], self.out_channels[-1])
            else:
                x = layer(x, self.up_transform[num_features - i])
        pred = self.heads(x)

        return pred

    def forward(self, x):
        z = self.backbone(x)
        pred = self.decoder(z)

        return pred


if __name__ == '__main__':
    import os
    from options.base_options import BaseOptions
    from utils.read import spiral_tramsform

    args = BaseOptions().parse()

    template_fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../template', 'template.ply')
    transform_fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../template', 'transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation)
    model = YTBHand(args, spiral_indices_list, up_transform_list)
    model.eval()
    img = torch.zeros([32, 3, 224, 224])
    res = model(img)
    print(res.size())

