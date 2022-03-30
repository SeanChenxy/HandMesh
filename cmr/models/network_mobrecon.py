import torch
import torch.nn as nn
from torch_scatter import scatter_add
from conv import DSConv


def Pool(x, trans, dim=1):
    row, col, value = trans[0].to(x.device), trans[1].to(x.device), trans[2].to(x.device)
    value = value.unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out2 = torch.zeros(x.size(0), row.size(0)//3, x.size(-1)).to(x.device)
    idx = row.unsqueeze(0).unsqueeze(-1).expand_as(out)
    out2 = torch.scatter_add(out2, dim, idx, out)
    return out2


class DWSpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(DWSpiralDeblock, self).__init__()
        self.conv = DSConv(in_channels, out_channels, indices)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = self.relu(self.conv(out))
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_chanel, kernel_size=3, padding=1, stride=1, dilation=1, relu=False, norm='bn'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_chanel, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_chanel)
        else:
            self.norm = None
        self.relu = nn.ReLU(True) if relu else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class ConvTBlock(nn.Module):
    def __init__(self, in_channel, out_chanel, kernel_size=3, padding=1, stride=2, output_padding=1, relu=False, norm='bn'):
        super(ConvTBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_chanel, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_chanel)
        else:
            self.norm = None
        self.relu = nn.ReLU(True) if relu else None

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class DWReg2DDecode3D(nn.Module):
    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel):
        super(DWReg2DDecode3D, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u[0].size(0)//3 for u in self.up_transform] + [self.up_transform[-1][0].size(0)//6]
        self.uv_channel = uv_channel

        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(DWSpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1]))
            else:
                self.de_layer.append(DWSpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1]))

        # head
        self.head = DSConv(self.out_channels[0], 3, self.spiral_indices[0])
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel])*0.01, requires_grad=True)

    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x):
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x = self.index(x, uv).permute(0, 2, 1)
        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x)
        num_features = len(self.de_layer)
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
        pred = self.head(x)

        return pred


def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, relu=True, group=1):
    _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation,
                      bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        sequence.append(nn.ReLU())

    return nn.Sequential(*sequence)


def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True):
    _linear = nn.Linear
    sequence = [_linear(channel_in, channel_out, bias=bias)]

    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        sequence.append(nn.Hardtanh(0,4))

    return nn.Sequential(*sequence)


class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, has_half_out=False, num3x3=1):
        # print('unit of mobile net block')
        super(mobile_unit, self).__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=1, padding=1, group=channel_in),
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.has_half_out = has_half_out

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = out + x
        if self.has_half_out:
            return half_out, out
        else:
            return out


class DenseBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in//4)
        self.conv2 = mobile_unit(channel_in*5//4, channel_in//4)
        self.conv3 = mobile_unit(channel_in*6//4, channel_in//4)
        self.conv4 = mobile_unit(channel_in*7//4, channel_in//4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2),dim=1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3),dim=1)
        out4 = self.conv4(comb3)
        comb4 = torch.cat((comb3, out4),dim=1)
        return comb4


class DenseBlock2(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in//2)
        self.conv2 = mobile_unit(channel_in*3//2, channel_in//2)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2),dim=1)
        return comb2


class DenseBlock3(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock3, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in)
        self.conv2 = mobile_unit(channel_in*2, channel_in)
        self.conv3 = mobile_unit(channel_in*3, channel_in)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2),dim=1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3),dim=1)
        return comb3


class DenseBlock2_noExpand(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2_noExpand, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in*3//4)
        self.conv2 = mobile_unit(channel_in*7//4, channel_in//4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((out1, out2),dim=1)
        return comb2


class SenetBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel, size):
        super(SenetBlock, self).__init__()
        self.size = size
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel = channel
        self.fc1 = linear_layer(self.channel, min(self.channel//2, 256))
        self.fc2 = linear_layer(min(self.channel//2, 256), self.channel, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_out = x
        pool = self.globalAvgPool(x)
        pool = pool.view(pool.size(0), -1)
        fc1 = self.fc1(pool)
        out = self.fc2(fc1)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)

        return out * original_out


class DenseStack(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel):
        super(DenseStack, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel*2, 32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel*2)
        self.senet2 = SenetBlock(input_channel*4,16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel*4)
        self.senet3 = SenetBlock(input_channel*8,8)
        self.transition3 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel*8)
        self.dense5 = DenseBlock2_noExpand(input_channel*8)
        self.thrink1 = nn.Sequential(mobile_unit(input_channel*8, input_channel*4, num3x3=1), mobile_unit(input_channel*4, input_channel*4, num3x3=2))
        self.senet4 = SenetBlock(input_channel*4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink2 = nn.Sequential(mobile_unit(input_channel*4, input_channel*2, num3x3=1), mobile_unit(input_channel*2, input_channel*2, num3x3=2))
        self.senet5 = SenetBlock(input_channel*2,8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink3 = nn.Sequential(mobile_unit(input_channel*2, input_channel*2, num3x3=1), mobile_unit(input_channel*2, output_channel, num3x3=2))
        self.senet6 = SenetBlock(output_channel,16)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.dense3(d2))
        u1 = self.upsample1(self.senet4(self.thrink1(d3)))
        us1 = d2 + u1
        u2 = self.upsample2(self.senet5(self.thrink2(us1)))
        us2 = d1 + u2
        u3 = self.upsample3(self.senet6(self.thrink3(us2)))
        return u3


class DenseStack2(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel, final_upsample=True, ret_mid=False):
        super(DenseStack2, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel*2,32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel*2)
        self.senet2 = SenetBlock(input_channel*4, 16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel*4)
        self.senet3 = SenetBlock(input_channel*8,8)
        self.transition3 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel*8)
        self.dense5 = DenseBlock2_noExpand(input_channel*8)
        self.thrink1 = nn.Sequential(mobile_unit(input_channel*8, input_channel*4, num3x3=1), mobile_unit(input_channel*4, input_channel*4, num3x3=2))
        self.senet4 = SenetBlock(input_channel*4,4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink2 = nn.Sequential(mobile_unit(input_channel*4, input_channel*2, num3x3=1), mobile_unit(input_channel*2, input_channel*2, num3x3=2))
        self.senet5 = SenetBlock(input_channel*2,8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink3 = nn.Sequential(mobile_unit(input_channel*2, input_channel*2, num3x3=1), mobile_unit(input_channel*2, output_channel, num3x3=2))
        self.senet6 = SenetBlock(output_channel,16)
        self.final_upsample = final_upsample
        if self.final_upsample:
            self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ret_mid = ret_mid

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.senet3(self.dense3(d2)))
        d4 = self.dense5(self.dense4(d3))
        u1 = self.upsample1(self.senet4(self.thrink1(d4)))
        us1 = d2 + u1
        u2 = self.upsample2(self.senet5(self.thrink2(us1)))
        us2 = d1 + u2
        u3 = self.senet6(self.thrink3(us2))
        if self.final_upsample:
            u3 = self.upsample3(u3)
        if self.ret_mid:
            return u3, u2, u1, d4
        else:
            return u3, d4


class Reorg(nn.Module):
    dump_patches = True

    def __init__(self):
        super(Reorg, self).__init__()

    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1], ss[2] // 2, 2, ss[3] // 2, 2). \
            permute(0, 1, 3, 5, 2, 4).contiguous().view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out