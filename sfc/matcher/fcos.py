# Modified according to repos https://github.com/tianzhi0549/FCOS, FCOS@tianzhi0549

import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator
from sfc.xcorr.xcorr import xcorr_depthwise

from sfc.layers import Scale


class SFHeader(torch.nn.Module):
    def __init__(self, cfg, in_channels, out_channels, kernel_size=3, centerness=False, scales=False):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(SFHeader, self).__init__()

        self.name = name
        self.is_centerness = centerness
        self.is_scales = scales
        
        # cls_tower = []
        kernel_tower = []
        search_tower = []
        for i in range(cfg.MODEL.SF.NUM_CONVS):
            kernel_tower.append(
                *nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        bias=False
                    ),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU6(inplace=True)
                )
            )

            search_tower.append(
                *nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        bias=False
                    ),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU6(inplace=True)
                )
            )
            
        self.kernel_conv = nn.Sequential(*kernel_tower)
        self.search_conv = nn.Sequential(*search_tower)

        self.header = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1,
            padding=1
        )
        
        if centerness:
            self.centerness = nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1
            )

        if centerness:
            modules = [self[name], self.header, self.centerness]
        else:
            modules = [self[name], self.header]

        for module in modules:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        if scales:
            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.cls_logits.bias, bias_value)

            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, kernels, searchs):
        outs = []
        if self.is_centerness:
            centerness = []

        for l, kernel, search in enumerate(zip(kernels, searchs)):
            k = self.kernel_conv(kernel)
            s = self.search_conv(search)
            o = xcorr_depthwise(s, k)

            if self.is_scales:
                outs.append(torch.exp(self.scales[l](self.header(o))))
            else:
                outs.append(self.header(o))

            if self.is_centerness:
                centerness.append(self.centerness(o))

        if self.is_scales:
            return outs
        else:
            return outs, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg):
        super(FCOSModule, self).__init__()

        # head = FCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, box_cls, box_regression, centerness, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class FCOS(nn.Module):
    def __init__(self, cfg, in_channels):
        self.cls = SFHeader(cfg, in_channels, 2)
        self.box_reg = SFHeader(cfg, in_channels, 4)
        self.calc_loss = FCOSModule(cfg)

    def forward(self, images, zx, sx, targets=None):
        box_cls, centerness = self.cls(zx, sx)
        box_regression = self.box_reg(zx, sx)
        loss = self.calc_loss(images, box_cls, box_regression, centerness, sx, targets)
        return loss


def build_matcher(cfg, in_channels):
    return FCOS(cfg, in_channels)
