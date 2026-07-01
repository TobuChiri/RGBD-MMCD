import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

from utils.engine.logger import get_logger
import warnings
import numpy as np

logger = get_logger()

def modality_drop(x_rgb, x_depth):
    modality_combination = [[1, 0], [0, 1], [1, 1]]
    index_list = [x for x in range(3)]

    p = []
    prob = np.array(( 1 / 3 , 1 / 3, 1 / 3))
    for i in range(x_rgb.shape[0]):
        index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
        p.append(modality_combination[index])
    p = np.array(p)
    p = torch.from_numpy(p)
    p = torch.unsqueeze(p, 2)
    p = torch.unsqueeze(p, 3)
    p = torch.unsqueeze(p, 4)
    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]
    x_depth = x_depth * p[:, 1]

    return x_rgb,x_depth,p

class EncoderDecoderOriginal(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='none', ignore_index=255), norm_layer=nn.BatchNorm2d, syncbn=False):
        super(EncoderDecoderOriginal, self).__init__()
        self.norm_layer = norm_layer
        self.cfg = cfg

        if cfg.backbone == 'DFormer-Large':
            from .encoders.DFormer import DFormer_Large as backbone
            self.channels=[96, 192, 288, 576]
        elif cfg.backbone == 'DFormer-Base':
            from .encoders.DFormer import DFormer_Base as backbone
            self.channels=[64, 128, 256, 512]
        elif cfg.backbone == 'DFormer-Small':
            from .encoders.DFormer import DFormer_Small as backbone
            self.channels=[64, 128, 256, 512]
        elif cfg.backbone == 'DFormer-Tiny':
            from .encoders.DFormer import DFormer_Tiny as backbone
            self.channels=[32, 64, 128, 256]

        if syncbn:
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        else:
            norm_cfg=dict(type='BN', requires_grad=True)

        if cfg.drop_path_rate is not None:
            self.backbone = backbone(drop_path_rate=cfg.drop_path_rate, norm_cfg = norm_cfg)
        else:
            self.backbone = backbone(drop_path_rate=0.1, norm_cfg = norm_cfg)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)

        elif cfg.decoder == 'ham':
            logger.info('Using Ham Decoder')
            print(cfg.num_classes)
            from .decoders.ham_head import LightHamHead as DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels[1:], num_classes=cfg.num_classes, in_index=[1,2,3],norm_cfg=norm_cfg, channels=cfg.decoder_embed_dim)
            from .decoders.fcnhead import FCNHead
            if cfg.aux_rate!=0:
                self.aux_index = 2
                self.aux_rate = cfg.aux_rate
                print('aux rate is set to',str(self.aux_rate))
                self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        elif cfg.decoder == 'nl':
            logger.info('Using Decoder: nl+')
            from .decoders.nl_head import NLHead as Head
            self.decode_head = Head(in_channels=self.channels[1:], in_index=[1,2,3],num_classes=cfg.num_classes, norm_cfg=norm_cfg,channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        if self.training:
            rgb,modal_x,p = modality_drop(rgb,modal_x)
        x = self.backbone(rgb, modal_x)

        if self.cfg.decoder == 'nl_near_far':
            out = self.decode_head.forward(x[0], modal_x=modal_x)
        else:
            out = self.decode_head.forward(x[0])
        out = F.interpolate(out, size=orisize[-2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[0][self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x=None, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        if label is not None:
            # 计算主损失
            mask = label.long() != self.cfg.background
            if mask.any():
                # 获取像素级损失
                pixel_loss = self.criterion(out, label.long())
                # 应用掩码并计算均值
                masked_loss = pixel_loss[mask]
                # 确保我们计算的是标量均值
                loss = masked_loss.mean() if masked_loss.numel() > 0 else torch.tensor(0.0, device=out.device)
            else:
                # 如果没有有效像素，返回零损失
                loss = torch.tensor(0.0, device=out.device)

            # 计算辅助损失（如果启用）
            if self.aux_head:
                if mask.any():
                    aux_pixel_loss = self.criterion(aux_fm, label.long())
                    aux_masked_loss = aux_pixel_loss[mask]
                    aux_loss = aux_masked_loss.mean() if aux_masked_loss.numel() > 0 else torch.tensor(0.0, device=out.device)
                    loss = loss + self.aux_rate * aux_loss
                # 如果mask为空，aux_loss已经是0，不需要额外处理

            # 确保返回的是标量
            if isinstance(loss, torch.Tensor):
                if loss.dim() > 0:
                    # 如果loss是多维张量，取均值
                    loss = loss.mean()
            return loss
        return out