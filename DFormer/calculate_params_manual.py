"""
手动计算DFormer模型参数量
基于代码分析进行计算
"""

def calculate_attention_params(dim, num_head, window=7):
    """计算attention模块参数量"""
    params = 0

    # Linear layers
    # q: dim -> dim
    params += dim * dim
    # q_cut: dim -> dim//2
    params += dim * (dim // 2)
    # a: dim -> dim
    params += dim * dim
    # l: dim -> dim
    params += dim * dim

    # Conv layers
    # conv: dim -> dim, kernel=7, groups=dim (depthwise)
    params += dim * 7 * 7  # depthwise conv

    # e_conv: dim//2 -> dim//2, kernel=7, groups=dim//2
    params += (dim // 2) * 7 * 7

    # e_fore: dim//2 -> dim//2
    params += (dim // 2) * (dim // 2)
    # e_back: dim//2 -> dim//2
    params += (dim // 2) * (dim // 2)

    # proj: dim//2*3 -> dim
    params += (dim // 2 * 3) * dim

    # proj_e: dim//2*3 -> dim//2
    params += (dim // 2 * 3) * (dim // 2)

    if window != 0:
        # short_cut_linear: dim//2*3 -> dim//2
        params += (dim // 2 * 3) * (dim // 2)
        # kv: dim -> dim
        params += dim * dim

    # LayerNorm参数 (weight和bias)
    params += 2 * dim  # norm
    params += 2 * (dim // 2)  # norm_e

    return params

def calculate_mlp_params(dim, mlp_ratio=4):
    """计算MLP模块参数量"""
    params = 0

    # fc1: dim -> dim * mlp_ratio
    params += dim * (dim * mlp_ratio)
    # fc2: dim * mlp_ratio -> dim
    params += (dim * mlp_ratio) * dim

    # pos conv: dim * mlp_ratio -> dim * mlp_ratio, kernel=3, groups=dim*mlp_ratio
    params += (dim * mlp_ratio) * 3 * 3  # depthwise conv

    # LayerNorm参数
    params += 2 * dim

    return params

def calculate_block_params(dim, num_head, mlp_ratio=4, window=7, drop_depth=False):
    """计算一个Block的参数量"""
    params = 0

    # attention模块
    params += calculate_attention_params(dim, num_head, window)

    # MLP模块
    params += calculate_mlp_params(dim, mlp_ratio)

    # layer scale参数
    params += dim  # layer_scale_1
    params += dim  # layer_scale_2

    if not drop_depth:
        params += dim // 2  # layer_scale_1_e
        params += dim // 2  # layer_scale_2_e
        # MLP for depth branch
        params += calculate_mlp_params(dim // 2, mlp_ratio)

    return params

def calculate_downsample_params(in_dim, out_dim, is_depth=False):
    """计算下采样层参数量"""
    params = 0

    if is_depth:
        # stem_e for depth branch
        # Conv1: 1 -> in_dim//4, kernel=3
        params += 1 * (in_dim // 4) * 3 * 3
        # Conv2: in_dim//4 -> in_dim//2, kernel=3
        params += (in_dim // 4) * (in_dim // 2) * 3 * 3

        # BN参数 (每个BN有2个参数: weight和bias)
        params += 2 * (in_dim // 4)
        params += 2 * (in_dim // 2)
    else:
        # stem for RGB branch
        # Conv1: 3 -> in_dim//2, kernel=3
        params += 3 * (in_dim // 2) * 3 * 3
        # Conv2: in_dim//2 -> in_dim, kernel=3
        params += (in_dim // 2) * in_dim * 3 * 3

        # BN参数
        params += 2 * (in_dim // 2)
        params += 2 * in_dim

    return params

def calculate_stage_downsample_params(in_dim, out_dim, is_depth=False):
    """计算stage间的下采样参数量"""
    params = 0

    # BN参数
    if is_depth:
        params += 2 * in_dim  # BN for depth
    else:
        params += 2 * in_dim  # BN for RGB

    # Conv: in_dim -> out_dim, kernel=3, stride=2
    if is_depth:
        params += in_dim * (out_dim // 2) * 3 * 3
    else:
        params += in_dim * out_dim * 3 * 3

    return params

def calculate_encoder_params(dims, depths, num_heads, mlp_ratios, windows):
    """计算整个encoder的参数量"""
    total_params = 0

    # 初始stem层
    total_params += calculate_downsample_params(dims[0], dims[0], is_depth=False)
    total_params += calculate_downsample_params(dims[0], dims[0], is_depth=True)

    # 各stage
    for i in range(len(dims)):
        # 当前stage的blocks
        for j in range(depths[i]):
            drop_depth = (i == 3) and (j == depths[i] - 1)  # 最后一个block的最后一个stage
            total_params += calculate_block_params(
                dim=dims[i],
                num_head=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                window=windows[i],
                drop_depth=drop_depth
            )

        # stage间的下采样（除了最后一个stage）
        if i < len(dims) - 1:
            total_params += calculate_stage_downsample_params(dims[i], dims[i+1], is_depth=False)
            total_params += calculate_stage_downsample_params(dims[i], dims[i+1], is_depth=True)

    return total_params

def main():
    print("=" * 80)
    print("DFormer Encoder 参数量计算")
    print("=" * 80)

    # 定义各版本配置
    versions = {
        'DFormer-Tiny': {
            'dims': [32, 64, 128, 256],
            'depths': [3, 3, 5, 2],
            'num_heads': [1, 2, 4, 8],
            'mlp_ratios': [8, 8, 4, 4],
            'windows': [0, 7, 7, 7]
        },
        'DFormer-Small': {
            'dims': [64, 128, 256, 512],
            'depths': [2, 2, 4, 2],
            'num_heads': [1, 2, 4, 8],
            'mlp_ratios': [8, 8, 4, 4],
            'windows': [0, 7, 7, 7]
        },
        'DFormer-Base': {
            'dims': [64, 128, 256, 512],
            'depths': [3, 3, 12, 2],
            'num_heads': [1, 2, 4, 8],
            'mlp_ratios': [8, 8, 4, 4],
            'windows': [0, 7, 7, 7]
        },
        'DFormer-Large': {
            'dims': [96, 192, 288, 576],
            'depths': [3, 3, 12, 2],
            'num_heads': [1, 2, 4, 8],
            'mlp_ratios': [8, 8, 4, 4],
            'windows': [0, 7, 7, 7]
        }
    }

    results = {}
    for name, config in versions.items():
        params = calculate_encoder_params(
            config['dims'],
            config['depths'],
            config['num_heads'],
            config['mlp_ratios'],
            config['windows']
        )
        results[name] = params

        print(f"\n{name}:")
        print(f"  配置: dims={config['dims']}, depths={config['depths']}")
        print(f"  计算参数量: {params:,}")
        print(f"  约: {params/1e6:.2f}M")

    # 估算decoder参数量（基于MLPDecoder）
    print("\n" + "=" * 80)
    print("Decoder 参数量估算 (MLPDecoder)")
    print("=" * 80)

    # MLPDecoder大致结构估算
    # 假设有4个输入通道，经过一些卷积和上采样操作
    # 这里给出一个粗略估算
    decoder_estimates = {
        'Tiny': 0.5e6,    # ~0.5M
        'Small': 1.0e6,   # ~1.0M
        'Base': 2.0e6,    # ~2.0M
        'Large': 3.0e6    # ~3.0M
    }

    print("\n总参数量估算 (Encoder + MLPDecoder):")
    for name in versions.keys():
        encoder_params = results[name]
        decoder_key = name.split('-')[1]
        decoder_est = decoder_estimates.get(decoder_key, 2.0e6)
        total_est = encoder_params + decoder_est

        print(f"\n{name}:")
        print(f"  Encoder: {encoder_params/1e6:.2f}M")
        print(f"  Decoder (估算): {decoder_est/1e6:.2f}M")
        print(f"  总计: {total_est/1e6:.2f}M")

    print("\n" + "=" * 80)
    print("注意：这是基于代码分析的估算值")
    print("实际值可能因具体实现细节有所不同")
    print("=" * 80)

if __name__ == "__main__":
    main()