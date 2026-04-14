import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

import models.configs_TransMorph_DSwin3D_AGDynConv as configs
from models.TransMorph import Conv3dReLU, DecoderBlock, RegistrationHead, SpatialTransformer


def _to_3tuple(x):
    if isinstance(x, int):
        return (x, x, x)
    return tuple(x)


def _split_heads(num_heads, num_branches):
    base = num_heads // num_branches
    remainder = num_heads % num_branches
    return [base + (1 if i < remainder else 0) for i in range(num_branches)]


def _build_relative_offsets(kernel_size, dilation):
    kz, ky, kx = _to_3tuple(kernel_size)
    dz, dy, dx = _to_3tuple(dilation)
    z_offsets = [(i - kz // 2) * dz for i in range(kz)]
    y_offsets = [(i - ky // 2) * dy for i in range(ky)]
    x_offsets = [(i - kx // 2) * dx for i in range(kx)]
    return [(x, y, z) for z in z_offsets for y in y_offsets for x in x_offsets]


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=4, in_chans=2, embed_dim=96, patch_norm=True):
        super().__init__()
        patch_size = _to_3tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = LayerNorm3d(embed_dim) if patch_norm else nn.Identity()

    def forward(self, x):
        depth, height, width = x.shape[2:]
        pad_d = (self.patch_size[0] - depth % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - height % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - width % self.patch_size[2]) % self.patch_size[2]
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = self.proj(x)
        return self.norm(x)


class PatchMerging3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduction = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.reduction(x)


class DeformableSlidingWindowAttentionGuided3D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        branch_kernel_sizes,
        branch_dilations,
        offset_kernel_size=3,
        offset_range_factor=1.0,
        qkv_bias=True,
        rel_pos_bias=True,
        use_pe=True,
        dwc_pe=True,
        proj_drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.offset_range_factor = offset_range_factor
        self.use_pe = use_pe
        self.dwc_pe = dwc_pe

        branch_kernel_sizes = [_to_3tuple(x) for x in branch_kernel_sizes]
        branch_dilations = [_to_3tuple(x) for x in branch_dilations]
        if len(branch_kernel_sizes) != len(branch_dilations):
            raise ValueError("branch_kernel_sizes and branch_dilations must have the same length")

        self.branch_head_splits = _split_heads(num_heads, len(branch_kernel_sizes))
        self.branch_offsets = [_build_relative_offsets(k, d) for k, d in zip(branch_kernel_sizes, branch_dilations)]

        self.q_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj_out = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        padding = offset_kernel_size // 2
        self.conv_offset = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=offset_kernel_size, padding=padding, groups=dim),
            LayerNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, num_heads * 3, kernel_size=1, bias=True),
        )

        self.branch_biases = nn.ParameterList()
        for branch_heads, offsets in zip(self.branch_head_splits, self.branch_offsets):
            init_bias = torch.zeros(branch_heads, len(offsets))
            self.branch_biases.append(nn.Parameter(init_bias, requires_grad=rel_pos_bias))

        self.rpe_table = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def _get_reference_grid(self, depth, height, width, batch_heads, dtype, device):
        z = torch.linspace(-1.0, 1.0, depth, dtype=dtype, device=device)
        y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
        x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        grid = torch.stack((xx, yy, zz), dim=-1)
        return grid.unsqueeze(0).expand(batch_heads, -1, -1, -1, -1)

    def _relative_grid_offset(self, offset_xyz, depth, height, width, dtype, device):
        dx, dy, dz = offset_xyz
        norm = torch.tensor(
            [
                2.0 / max(width - 1, 1),
                2.0 / max(height - 1, 1),
                2.0 / max(depth - 1, 1),
            ],
            dtype=dtype,
            device=device,
        )
        base = torch.tensor([dx, dy, dz], dtype=dtype, device=device)
        return (base * norm).view(1, 1, 1, 1, 3)

    def _branch_attention(self, q, k, v, offsets, relative_offsets, branch_bias):
        batch_size, branch_heads, head_dim, depth, height, width = q.shape
        batch_heads = batch_size * branch_heads

        q = q.reshape(batch_heads, head_dim, depth, height, width)
        k = k.reshape(batch_heads, head_dim, depth, height, width)
        v = v.reshape(batch_heads, head_dim, depth, height, width)
        offsets = offsets.reshape(batch_heads, depth, height, width, 3)

        reference = self._get_reference_grid(depth, height, width, batch_heads, q.dtype, q.device)

        score_list = []
        rel_grids = []
        for rel_index, rel_offset in enumerate(relative_offsets):
            rel_grid = self._relative_grid_offset(rel_offset, depth, height, width, q.dtype, q.device)
            rel_grids.append(rel_grid)
            sample_grid = (reference + offsets + rel_grid).clamp(-1.0, 1.0)
            k_sample = F.grid_sample(
                k,
                sample_grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True,
            )
            score = (q * k_sample).sum(dim=1) * self.scale
            score = score.reshape(batch_size, branch_heads, depth, height, width)
            score = score + branch_bias[:, rel_index].view(1, branch_heads, 1, 1, 1)
            score_list.append(score)

        attn = torch.stack(score_list, dim=2)
        attn = self.attn_drop(attn.softmax(dim=2))

        out = torch.zeros_like(q).reshape(batch_size, branch_heads, head_dim, depth, height, width)
        for rel_index, rel_grid in enumerate(rel_grids):
            sample_grid = (reference + offsets + rel_grid).clamp(-1.0, 1.0)
            v_sample = F.grid_sample(
                v,
                sample_grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True,
            ).reshape(batch_size, branch_heads, head_dim, depth, height, width)
            weight = attn[:, :, rel_index].unsqueeze(2)
            out = out + weight * v_sample

        return out

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        residual_lepe = self.rpe_table(q)

        offsets = self.conv_offset(q)
        offsets = offsets.reshape(batch_size, self.num_heads, 3, depth, height, width)
        offsets = offsets.permute(0, 1, 3, 4, 5, 2)

        offset_scale = torch.tensor(
            [
                2.0 / max(width - 1, 1),
                2.0 / max(height - 1, 1),
                2.0 / max(depth - 1, 1),
            ],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 1, 1, 1, 3)
        offsets = offsets.tanh() * offset_scale * self.offset_range_factor

        q = q.reshape(batch_size, self.num_heads, self.head_dim, depth, height, width)
        k = k.reshape(batch_size, self.num_heads, self.head_dim, depth, height, width)
        v = v.reshape(batch_size, self.num_heads, self.head_dim, depth, height, width)

        q_splits = torch.split(q, self.branch_head_splits, dim=1)
        k_splits = torch.split(k, self.branch_head_splits, dim=1)
        v_splits = torch.split(v, self.branch_head_splits, dim=1)
        offset_splits = torch.split(offsets, self.branch_head_splits, dim=1)

        branch_outputs = []
        branch_energies = []
        for q_branch, k_branch, v_branch, offset_branch, rel_offsets, bias in zip(
            q_splits,
            k_splits,
            v_splits,
            offset_splits,
            self.branch_offsets,
            self.branch_biases,
        ):
            branch_out = self._branch_attention(q_branch, k_branch, v_branch, offset_branch, rel_offsets, bias)
            branch_outputs.append(branch_out)
            branch_energy = branch_out.abs().mean(dim=(1, 2, 3, 4, 5), keepdim=False).unsqueeze(1)
            branch_energies.append(branch_energy)

        out = torch.cat(branch_outputs, dim=1).reshape(batch_size, channels, depth, height, width)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe

        branch_energy = torch.cat(branch_energies, dim=1)
        offset_magnitude = torch.linalg.vector_norm(offsets, dim=-1).mean(dim=(2, 3, 4))
        guide_vector = torch.cat([offset_magnitude, branch_energy], dim=1)
        aux = {
            'offsets': offsets,
            'offset_magnitude': offset_magnitude,
            'branch_energy': branch_energy,
            'guide_vector': guide_vector,
        }

        out = self.proj_drop(self.proj_out(out))
        return out, aux


class AttentionGuidedDynamicRangeDWConv3D(nn.Module):
    def __init__(
        self,
        channels,
        guide_dim,
        branch_kernel_sizes,
        branch_dilations,
        reduction=4,
        bias=True,
        detach_guidance=True,
    ):
        super().__init__()
        branch_kernel_sizes = [_to_3tuple(k) for k in branch_kernel_sizes]
        branch_dilations = [_to_3tuple(d) for d in branch_dilations]
        if len(branch_kernel_sizes) != len(branch_dilations):
            raise ValueError("branch_kernel_sizes and branch_dilations must have the same length")

        self.detach_guidance = detach_guidance
        self.guide_dim = guide_dim
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
        self.gate_norm = nn.LayerNorm(channels + guide_dim)
        self.gate = nn.Sequential(
            nn.Linear(channels + guide_dim, gate_hidden, bias=True),
            nn.GELU(),
            nn.Linear(gate_hidden, len(self.branches), bias=True),
        )

    def forward(self, x, guidance):
        feat_pool = F.adaptive_avg_pool3d(x, 1).flatten(1)
        if guidance is None:
            guidance = feat_pool.new_zeros(feat_pool.size(0), self.guide_dim)
        if self.detach_guidance:
            guidance = guidance.detach()

        gate_input = torch.cat([feat_pool, guidance], dim=1)
        gate_input = self.gate_norm(gate_input)
        weights = torch.softmax(self.gate(gate_input), dim=1)

        out = 0.0
        for branch_index, branch in enumerate(self.branches):
            branch_out = branch(x)
            alpha = weights[:, branch_index].view(-1, 1, 1, 1, 1)
            out = out + alpha * branch_out
        return out


class FeedForwardAttentionGuidedDynConv3D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_attn_branches,
        expansion_factor=2.0,
        branch_kernel_sizes=((3, 3, 3), (3, 3, 3), (1, 5, 5)),
        branch_dilations=((1, 1, 1), (2, 2, 2), (1, 1, 1)),
        gate_reduction=4,
        bias=True,
        detach_guidance=True,
    ):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dynamic_dwconv = AttentionGuidedDynamicRangeDWConv3D(
            hidden * 2,
            guide_dim=num_heads + num_attn_branches,
            branch_kernel_sizes=branch_kernel_sizes,
            branch_dilations=branch_dilations,
            reduction=gate_reduction,
            bias=bias,
            detach_guidance=detach_guidance,
        )
        self.project_out = nn.Conv3d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x, guidance):
        x = self.project_in(x)
        x1, x2 = self.dynamic_dwconv(x, guidance).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class DSwinTransformerBlockAGDynConv3D(nn.Module):
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
        detach_guidance=True,
        qkv_bias=True,
        rel_pos_bias=True,
        use_pe=True,
        dwc_pe=True,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm3d(dim)
        self.attn = DeformableSlidingWindowAttentionGuided3D(
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
        self.ffn = FeedForwardAttentionGuidedDynConv3D(
            dim=dim,
            num_heads=num_heads,
            num_attn_branches=len(branch_kernel_sizes),
            expansion_factor=ffn_expansion_factor,
            branch_kernel_sizes=ffn_branch_kernel_sizes,
            branch_dilations=ffn_branch_dilations,
            gate_reduction=ffn_gate_reduction,
            bias=qkv_bias,
            detach_guidance=detach_guidance,
        )

    def forward(self, x):
        attn_out, attn_aux = self.attn(self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ffn(self.norm2(x), attn_aux['guide_vector']))
        return x, attn_aux


class DSwinStageAGDynConv3D(nn.Module):
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
        detach_guidance,
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
                DSwinTransformerBlockAGDynConv3D(
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
                    detach_guidance=detach_guidance,
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
        stage_aux = []
        for block in self.blocks:
            x, attn_aux = block(x)
            stage_aux.append(attn_aux)
        out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return out, x, stage_aux


class DSwinEncoderAGDynConv3D(nn.Module):
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
                DSwinStageAGDynConv3D(
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
                    detach_guidance=config.detach_guidance,
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
        aux = []
        for stage in self.stages:
            out, x, stage_aux = stage(x)
            outs.append(out)
            aux.extend(stage_aux)
        return outs, aux


class TransMorphDSwin3DAGDynConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim

        self.encoder = DSwinEncoderAGDynConv3D(config)
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

        out_feats, attn_aux = self.encoder(x)

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
        return out, flow, {'attn_aux': attn_aux}


def compute_offset_magnitude_loss(attn_aux_list, penalty='l2'):
    if not attn_aux_list:
        raise ValueError("attn_aux_list must not be empty")

    losses = []
    for aux in attn_aux_list:
        offsets = aux['offsets']
        magnitude = torch.linalg.vector_norm(offsets, dim=-1)
        if penalty == 'l2':
            magnitude = magnitude * magnitude
        elif penalty != 'l1':
            raise ValueError(f"Unsupported penalty: {penalty}")
        losses.append(magnitude.mean())
    return sum(losses) / len(losses)


def compute_offset_smoothness_loss(attn_aux_list, penalty='l2'):
    if not attn_aux_list:
        raise ValueError("attn_aux_list must not be empty")

    losses = []
    for aux in attn_aux_list:
        offsets = aux['offsets']
        dd = torch.abs(offsets[:, :, 1:, :, :, :] - offsets[:, :, :-1, :, :, :])
        dh = torch.abs(offsets[:, :, :, 1:, :, :] - offsets[:, :, :, :-1, :, :])
        dw = torch.abs(offsets[:, :, :, :, 1:, :] - offsets[:, :, :, :, :-1, :])
        if penalty == 'l2':
            dd = dd * dd
            dh = dh * dh
            dw = dw * dw
        elif penalty != 'l1':
            raise ValueError(f"Unsupported penalty: {penalty}")
        losses.append((dd.mean() + dh.mean() + dw.mean()) / 3.0)
    return sum(losses) / len(losses)


CONFIGS = {
    'TransMorph-DSwin3D-AGDynConv': configs.get_3DTransMorphDSwinAGDynConv_config(),
}
