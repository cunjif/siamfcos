from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.fcos.fcos import build_fcos

def get_fcos(cfg, in_channels):
    return build_fcos(cfg, in_channels)