from __future__ import (
    division, 
    absolute_import, 
    print_function,
    unicode_literals
)

from torch import nn
from sfc.backbone import build_backbone
from sfc.adjust_layers import build_neck
from sfc.matcher import build_matcher


class SiamTracker(nn.Module):
    def __init__(self):
        super(SiamTracker, self).__init__()  

        self.backbone = build_backbone(cfg)

        if cfg.MODEL.NECK.USE:
            self.neck = build_neck(cfg, self.backbone.output_channels)

        self.matcher = build_matcher(cfg, self.neck.output_channels)
        # self.roi_header(build_match)

    def forward(self, x, targets):
        pass

def build_model(cfg):
    return SiamTracker(cfg)