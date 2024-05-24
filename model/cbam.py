import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    """Implementation of Convolutional Block Attention Module (CBAM) as described in Woo et al., 2018"""
    def __init__(self, n_channels_in, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        spat_att = self.spatial_attention(fp)
        fpp = spat_att * fp
        return fpp


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = max(1, self.n_channels_in // self.reduction_ratio)

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck
        sig_pool = torch.sigmoid(pool_sum).unsqueeze(2).unsqueeze(3)

        return sig_pool


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        att = torch.sigmoid(conv)
        return att
