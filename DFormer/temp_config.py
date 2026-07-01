
class C:
    backbone = "DFormer-Base"
    num_classes = 40
    background = 0
    decoder = "MLPDecoder"
    decoder_embed_dim = 768
    drop_path_rate = 0.1
    pretrained_model = None
    bn_eps = 1e-5
    bn_momentum = 0.1
    use_dual_backbone = True  # 会在测试中动态修改
