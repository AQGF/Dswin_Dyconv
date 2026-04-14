import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

import models.configs_TransMorph_DSwin3D_DynConv as configs
from models.TransMorph import Conv3dReLU, DecoderBlock, RegistrationHead, SpatialTransformer
from models.TransMorph_DSwin3D import (
    DeformableSlidingWindowAttention3D,
    LayerNorm3d,
    PatchEmbed3D,
    PatchMerging3D,
)


def _to_3tuple(x):
    if isinstance(x, int):
        return (x, x, x)
    return tuple(x)


class DynamicRangeDWConv3D(nn.Module):
    def __init__(self, channels, branch_kernel_sizes, branch_dilations, reduction=4, bias=True):
        super().__init__()
        branch_kernel_sizes = [_to_3tuple(k) for k in branch_kernel_sizes]
        branch_dilations = [_to_3tuple(d) for d in branch_dilations]
        if len(branch_kernel_sizes) != len(branch_dilations):
            raise ValueError("branch_kernel_sizes and branch_dilations must have the same length")

        self.branches = nn.ModuleList()
        for kernel_size, dilation in zip(branch_kernel_sizes, branch_dilations):
            padding = tuple(((k - 1) * d) // 2 for k, d in zip(kernel_size, dilation))
            self.branches.append(
                nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    groups=channels,
                    bias=bias,
                )
            )

        gate_hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, gate_hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(gate_hidden, len(self.branches), kernel_size=1, bias=True),
        )

    def forward(self, x):
        weights = self.gate(x).flatten(1)
        weights = torch.softmax(weights, dim=1)

        out = 0
        for branch_index, branch in enumerate(self.branches):
            branch_out = branch(x)
            alpha = weights[:, branch_index].view(-1, 1, 1, 1, 1)
            out = out + alpha * branch_out
        return out


class FeedForwardDynConv3D(nn.Module):
    def __init__(
        self,
        dim,
        expansion_factor=2.0,
        branch_kernel_sizes=((3, 3, 3), (3, 3, 3), (1, 5, 5)),
        branch_dilations=((1, 1, 1), (2, 2, 2), (1, 1, 1)),
        gate_reduction=4,
        bias=True,
    ):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dynamic_dwconv = DynamicRangeDWConv3D(
            hidden * 2,
            branch_kernel_sizes=branch_kernel_sizes,
            branch_dilations=branch_dilations,
            reduction=gate_reduction,
            bias=bias,
        )
        self.project_out = nn.Conv3d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dynamic_dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class DSwinTransformerBlockDynConv3D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        branch_kernel_sizes,
        branch_dilations,
        offset_kernel_size=3,
        offset_range_factor=1.0,
        ffn_expansion_factor=2.0,
        ffn_branch_kernel_sizes=((3, 3, 3), (3, 3, 3), (1, 5, 5)),
        ffn_branch_dilations=((1, 1, 1), (2, 2, 2), (1, 1, 1)),
        ffn_gate_reduction=4,
        qkv_bias=True,
        rel_pos_bias=True,
        use_pe=True,
        dwc_pe=True,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm3d(dim)
        self.attn = DeformableSlidingWindowAttention3D(
            dim=dim,
            num_heads=num_heads,
            branch_kernel_sizes=branch_kernel_sizes,
            branch_dilations=branch_dilations,
            offset_kernel_size=offset_kernel_size,
            offset_range_factor=offset_range_factor,
            qkv_bias=qkv_bias,
            rel_pos_bias=rel_pos_bias,
            use_pe=use_pe,
            dwc_pe=dwc_pe,
            proj_drop=drop,
            attn_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = LayerNorm3d(dim)
        self.ffn = FeedForwardDynConv3D(
            dim,
            expansion_factor=ffn_expansion_factor,
            branch_kernel_sizes=ffn_branch_kernel_sizes,
            branch_dilations=ffn_branch_dilations,
            gate_reduction=ffn_gate_reduction,
            bias=qkv_bias,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class DSwinStageDynConv3D(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        branch_kernel_sizes,
        branch_dilations,
        offset_kernel_size,
        offset_range_factor,
        ffn_expansion_factor,
        ffn_branch_kernel_sizes,
        ffn_branch_dilations,
        ffn_gate_reduction,
        qkv_bias,
        rel_pos_bias,
        use_pe,
        dwc_pe,
        drop,
        drop_path_rates,
        downsample=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DSwinTransformerBlockDynConv3D(
                    dim=dim,
                    num_heads=num_heads,
                    branch_kernel_sizes=branch_kernel_sizes,
                    branch_dilations=branch_dilations,
                    offset_kernel_size=offset_kernel_size,
                    offset_range_factor=offset_range_factor,
                    ffn_expansion_factor=ffn_expansion_factor,
                    ffn_branch_kernel_sizes=ffn_branch_kernel_sizes,
                    ffn_branch_dilations=ffn_branch_dilations,
                    ffn_gate_reduction=ffn_gate_reduction,
                    qkv_bias=qkv_bias,
                    rel_pos_bias=rel_pos_bias,
                    use_pe=use_pe,
                    dwc_pe=dwc_pe,
                    drop=drop,
                    drop_path=drop_path_rates[i],
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging3D(dim, dim * 2) if downsample else None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return out, x


class DSwinEncoderDynConv3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            patch_norm=config.patch_norm,
        )

        stage_dims = [config.embed_dim * (2 ** i) for i in range(len(config.depths))]
        dpr = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()

        self.stages = nn.ModuleList()
        offset = 0
        for stage_index, (dim, depth, heads) in enumerate(zip(stage_dims, config.depths, config.num_heads)):
            stage_drop = dpr[offset:offset + depth]
            offset += depth
            self.stages.append(
                DSwinStageDynConv3D(
                    dim=dim,
                    depth=depth,
                    num_heads=heads,
                    branch_kernel_sizes=config.branch_kernel_sizes[stage_index],
                    branch_dilations=config.branch_dilations[stage_index],
                    offset_kernel_size=config.offset_kernel_size,
                    offset_range_factor=config.offset_range_factor,
                    ffn_expansion_factor=config.ffn_expansion_factor,
                    ffn_branch_kernel_sizes=config.ffn_branch_kernel_sizes,
                    ffn_branch_dilations=config.ffn_branch_dilations,
                    ffn_gate_reduction=config.ffn_gate_reduction,
                    qkv_bias=config.qkv_bias,
                    rel_pos_bias=config.rel_pos_bias,
                    use_pe=config.use_pe,
                    dwc_pe=config.dwc_pe,
                    drop=config.drop_rate,
                    drop_path_rates=stage_drop,
                    downsample=stage_index < len(config.depths) - 1,
                )
            )

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        for stage in self.stages:
            out, x = stage(x)
            outs.append(out)
        return outs


class TransMorphDSwin3DDynConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim

        self.encoder = DSwinEncoderDynConv3D(config)
        self.up0 = DecoderBlock(embed_dim * 8, embed_dim * 4, skip_channels=embed_dim * 4 if self.if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if self.if_transskip else 0, use_batchnorm=False)
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if self.if_transskip else 0, use_batchnorm=False)
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if self.if_convskip else 0, use_batchnorm=False)
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan, skip_channels=config.reg_head_chan if self.if_convskip else 0, use_batchnorm=False)

        self.c1 = Conv3dReLU(config.in_chans, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(config.in_chans, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(config.reg_head_chan, 3, kernel_size=3)
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        source = x[:, 0:1, ...]

        if self.if_convskip:
            x_s0 = x
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

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
        x = self.up3(x, f4)
        x = self.up4(x, f5)

        flow = self.reg_head(x)
        out = self.spatial_trans(source, flow)
        return out, flow


CONFIGS = {
    'TransMorph-DSwin3D-DynConv': configs.get_3DTransMorphDSwinDynConv_config(),
    'TransMorph-DSwin3D-DynConv-Tiny': configs.get_3DTransMorphDSwinDynConvTiny_config(),
}
