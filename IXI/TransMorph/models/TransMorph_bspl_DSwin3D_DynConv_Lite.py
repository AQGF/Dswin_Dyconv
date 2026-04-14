import math
import torch
from torch import nn as nn
from torch.nn import functional as F

import models.transformation as transformation
import models.TransMorph as TM
import models.configs_TransMorph_bspl_DSwin3D_DynConv_Lite as configs
from models.TransMorph_DSwin3D_DynConv import DSwinEncoderDynConv3D


def convNd(ndim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, a=0.0):
    conv_nd = getattr(nn, f"Conv{ndim}d")(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    nn.init.kaiming_uniform_(conv_nd.weight, a=a)
    return conv_nd


def interpolate_(x, scale_factor=None, size=None, mode=None):
    if mode == 'nearest':
        interp_mode = mode
    else:
        ndim = x.ndim - 2
        if ndim == 1:
            interp_mode = 'linear'
        elif ndim == 2:
            interp_mode = 'bilinear'
        elif ndim == 3:
            interp_mode = 'trilinear'
        else:
            raise ValueError(f'Data dimension ({ndim}) must be 2 or 3')
    return F.interpolate(x, scale_factor=scale_factor, size=size, mode=interp_mode)


class TranMorphBSplineDSwinDynConvLiteNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        ndim = 3
        img_size = config.img_size
        cps = config.cps
        resize_channels = config.resize_channels
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")
        self.output_size = tuple([int(math.ceil((imsz - 1) / c) + 1 + 2) for imsz, c in zip(img_size, cps)])

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim

        self.encoder = DSwinEncoderDynConv3D(config)
        self.up0 = TM.DecoderBlock(embed_dim * 8, embed_dim * 4, skip_channels=embed_dim * 4 if self.if_transskip else 0, use_batchnorm=False)
        self.up1 = TM.DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if self.if_transskip else 0, use_batchnorm=False)
        self.up2 = TM.DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if self.if_transskip else 0, use_batchnorm=False)
        self.up3 = TM.DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if self.if_convskip else 0, use_batchnorm=False)
        self.c1 = TM.Conv3dReLU(config.in_chans, embed_dim // 2, 3, 1, use_batchnorm=False)

        self.resize_conv = nn.ModuleList()
        for i in range(len(resize_channels)):
            in_ch = embed_dim // 2 if i == 0 else resize_channels[i - 1]
            out_ch = resize_channels[i]
            self.resize_conv.append(nn.Sequential(convNd(ndim, in_ch, out_ch, a=0.2), nn.LeakyReLU(0.2)))

        self.out_layer = convNd(ndim, resize_channels[-1], ndim)
        self.transform = transformation.CubicBSplineFFDTransform(ndim=3, svf=True, cps=cps)

    def forward(self, inputs):
        src, tar = inputs
        x = torch.cat((src, tar), dim=1)

        if self.if_convskip:
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
        else:
            f4 = None

        out_feats = self.encoder(x)
        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None

        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        dec_out = self.up3(x, f4)

        x = interpolate_(dec_out, size=self.output_size)
        for resize_layer in self.resize_conv:
            x = resize_layer(x)
        x = self.out_layer(x)
        flow, disp = self.transform(x)
        y = transformation.warp(src, disp)
        return y, flow, disp


CONFIGS = {
    'TransMorphBSpline-DSwin3D-DynConv-Lite': configs.get_TransMorphBsplDSwinDynConvLite_config(),
}
