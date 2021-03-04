import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from conv import SpiralConv


def Pool(x, trans, dim=1):
    """
    :param x: input feature
    :param trans: upsample matrix
    :param dim: upsample dimension
    :return: upsampled feature
    """
    trans = trans.to(x.device)
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class ParallelDeblock(nn.Module):
    """
    ISM in the paper. Note that "indices[:, :indices.size(1)//3]" is approximate and not-strict k-disk.
    In this way, a from-small-to-large receptive field can be obtained, and "conv" can be shared among all vertices.
    """
    def __init__(self, in_channels, out_channels, indices):
        super(ParallelDeblock, self).__init__()
        indices_2d3 = indices[:, :indices.size(1)//3*2]
        indices_d3 = indices[:, :indices.size(1)//3]
        indices_1 = indices[:, 0:1]
        self.conv_2d3 = SpiralConv(in_channels, out_channels // 4, indices_2d3)
        self.conv_d3 = SpiralConv(in_channels, out_channels // 4, indices_d3)
        self.conv = SpiralConv(in_channels, out_channels // 2, indices)
        self.conv1 = SpiralConv(in_channels, out_channels, indices_1)

    def forward(self, x, up_transform):
        """
        :param x: input feature
        :param up_transform: upsample matrix
        :return: upsampled and convoluted feature
        """
        out = Pool(x, up_transform)
        short_cut = self.conv1(out)
        p_d3 = self.conv_d3(out)
        p_2d3 = self.conv_2d3(out)
        p = self.conv(out)

        f = torch.cat((p, p_2d3, p_d3), 2)
        out = F.relu(short_cut + f)

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


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Linear(in_dim, in_dim)
        self.key_conv = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)

        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, num_dim = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, 1, num_dim).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, 1, num_dim)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, 1, num_dim)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, num_dim)

        out = self.gamma * out + x

        return out
