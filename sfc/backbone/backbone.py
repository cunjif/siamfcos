# Modified according to repos https://github.com/tianzhi0549/FCOS, FCOS@tianzhi0549

from collections import OrderedDict
from torch import nn
from sfc.utils import registry
from sfc.layers import conv_with_kaiming_uniform
from . import resnet
from . import fpn as fpn_module


@registry.BACKBONE_REGISTRY.register("R-50-FPN")
@registry.BACKBONE_REGISTRY.register("R-101-FPN")
@registry.BACKBONE_REGISTRY.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNET.RES2_OUTPUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE_OUTPUT_CHANNELS

    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        output_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MDOEL.FPN.USE_GN,
            cfg.MDOEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool()
    )
    
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.output_channels = out_channels
    return model


@registry.BACKBONE_REGISTRY.register("R-50-FPN-RETINANET")
@registry.BACKBONE_REGISTRY.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNET.RES2_OUTPUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE_OUTPUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 4 if cfg.MODEL.RETINANET.USE_C5 else out_channels
    
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ], 
        out_channels=out_channels, 
        conv_block=conv_with_kaiming_uniform(cfg.MODEL.FPN.USE_GN, 
        cfg.MODEL.FPN.USE_RELU),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels)
    )

    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model



def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.BODY in registry.BACKBONE_REGISTRY, \
        f"Backbone in cfg.MODEL.BACKBONE.BODY: {cfg.MODEL.BACKBONE.BODY} is not regist"

    return registry.BACKBONE_REGISTRY[cfg.MODEL.BACKBONE.BODY]