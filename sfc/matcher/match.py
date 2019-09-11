from sfc.matcher import build_fcos


def build_matcher(cfg, in_channels):
    return build_fcos(cfg, in_channels)