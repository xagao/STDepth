# Copyright Niantic 2019. Patent Pending. All rights reserved.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from collections import OrderedDict
from layers import *


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by RELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = Conv3x3(in_channels, out_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlin(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlin(out)
        return out
    
class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

def upsample_b(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)


class DF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DF, self).__init__()
        self.resblock1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.resblock2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.embedding = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, skip):
        Sadd = skip + x
        attn = self.resblock1(x)
        attn = self.embedding(attn)
        attn = attn.view(attn.size(0), attn.size(1), -1)
        attn = self.softmax(attn)
        attn = attn.view(x.size(0), x.size(1), x.size(2), x.size(3))
        Scaled = attn * Sadd
        out = self.resblock2(Scaled)
        out = out + Sadd
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv1(out)
        return out
    
    
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class GFR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(GFR, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2
        #self.conv1 = nn.Conv2d(in_planes * 2, in_planes, kernel_size=1)
        
        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)
        
        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, 1, H*W]
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

# parallel
    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out

# concat
    # def forward(self, x):
    #     # [N, C, H, W]
    #     out = self.spatial_pool(x)
    #     # [N, C, H, W]
    #     out = self.channel_pool(out)
    #     # [N, C, H, W]
    #     # out = context_spatial + context_channel
    #     return out
    
    
class DepthDecoder_Mix(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder_Mix, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        
        "mpvit_xs"
        self.num_ch_dec = np.array([32, 64, 128, 192, 256])
        
        # num_ch_enc = np.array([64, 128, 216, 288, 288])
        self.num_ch_dec_0 = np.array([16])
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # conv1x1 在每个encoder特征之后
            num_ch_in = self.num_ch_enc[i]
            num_ch_out = self.num_ch_enc[i]  
            self.convs[("conv1x1", i, 0)] = nn.Conv2d(num_ch_in, num_ch_out, kernel_size=1)
            
            # sc_attention
            if i == 4:
                self.convs[("gfr", i)] = GFR(num_ch_in, num_ch_out)
            elif i > 1 and i < 4:
                self.convs[("Residual", i, 0)] = ConvBlock(num_ch_in * 2, num_ch_out)
                self.convs[("Residual", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
                self.convs[("gfr", i)] = GFR(num_ch_in, num_ch_out)
            elif i < 2:
                num_ch_out = self.num_ch_dec[i]
                self.convs[("df", i)] = DF(num_ch_in, num_ch_out)
            
            if i >= 2:
                num_ch_in = self.num_ch_enc[i] 
                num_ch_out = self.num_ch_dec[i] 
                self.convs[("conv1x1", i, 1)] = Conv1x1(num_ch_in, num_ch_out)
            
            
        num_ch_in = self.num_ch_dec[0]
        num_ch_out =  self.num_ch_dec_0[0]
        self.convs[("conv1", 0)] = Conv1x1(num_ch_in, num_ch_out)
        
        for s in self.scales:
            if s > 0:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            elif s == 0:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec_0[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        device = input_features[0].device
        
        device_of_conv = self.convs[("conv1x1", 1, 0)].weight.device

        # decoder
        x = input_features[-1]

        for i in range(4, -1, -1):
            if i == 4:
                x = self.convs[("conv1x1", i, 0)](x)
                x = self.convs[("gfr", i)](x)
                x = upsample_b(x)
                x = self.convs[("conv1x1", i, 1)](x)
            elif i > 1 and i < 4:
                x1 = input_features[i]
                x1 = self.convs[("conv1x1", i, 0)](x1)
                x = torch.cat([x, x1], dim=1)
                x = self.convs[("Residual", i, 0)](x)
                x = self.convs[("gfr", i)](x)
                x = self.convs[("Residual", i, 1)](x)
                x = upsample_b(x)
                x = self.convs[("conv1x1", i, 1)](x)
                #self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
            elif i <= 1:
                skip = input_features[i]
                skip = self.convs[("conv1x1", i, 0)](skip)
                x = self.convs[("df", i)](x, skip)
                if i == 0:
                    x = self.convs[("conv1", 0)](x)
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                    
        return self.outputs

