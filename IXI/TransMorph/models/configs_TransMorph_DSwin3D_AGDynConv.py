import ml_collections


def get_3DTransMorphDSwinAGDynConv_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.branch_kernel_sizes = (
        ((3, 3, 3), (3, 3, 3), (1, 5, 5)),
        ((3, 3, 3), (3, 3, 3), (3, 5, 5)),
        ((3, 3, 3), (3, 3, 3), (5, 5, 5)),
        ((3, 3, 3), (3, 3, 3), (5, 5, 5)),
    )
    config.branch_dilations = (
        ((1, 1, 1), (2, 2, 2), (1, 1, 1)),
        ((1, 1, 1), (2, 2, 2), (1, 1, 1)),
        ((1, 1, 1), (2, 2, 2), (1, 1, 1)),
        ((1, 1, 1), (2, 2, 2), (1, 1, 1)),
    )
    config.offset_kernel_size = 3
    config.offset_range_factor = 1.0
    config.ffn_expansion_factor = 2.0
    config.ffn_branch_kernel_sizes = ((3, 3, 3), (3, 3, 3), (1, 5, 5))
    config.ffn_branch_dilations = ((1, 1, 1), (2, 2, 2), (1, 1, 1))
    config.ffn_gate_reduction = 4
    config.detach_guidance = True
    config.qkv_bias = False
    config.drop_rate = 0.0
    config.drop_path_rate = 0.3
    config.use_pe = True
    config.dwc_pe = True
    config.rel_pos_bias = True
    config.patch_norm = True
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.offset_mag_weight = 1.0
    config.offset_smooth_weight = 1.0
    config.offset_reg_penalty = 'l2'
    return config
