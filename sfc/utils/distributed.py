import torch.distributed as dist
import torch
from torch import nn

def reduce_gradients():
    pass


def average_reduce():
    pass

def dist_init():
    return get_rank(), get_world_size()


def get_rank():
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_rank()


def broadcast_parameters(model):
    for p in model.state_dict().values():
        dist.broadcast(p, 0)


def broadcast_buffers(model, method=0):
    if method == 0:
        return

    world_size = get_world_size()

    for b in model._all_buffers():
        if method == 1:
            dist.broadcast(b, 0)
        elif method == 2:
            dist.all_reduce(b)
            b /= world_size
        else:
            raise Exception(f"Not implemented method {method}")


class DistModule(nn.Module):
    def __init__(self, module, bn_method=0):
        super(DistModule, self).__init__()
        self.module = module
        self.bn_method = bn_method
        if get_world_size() > 1:
            broadcast_parameters(self.module)
        else:
            self.bn_method = 0
    
    def forward(self, *args, **kwargs):
        broadcast_buffers(self.module, self.bn_method)
        return self.module(*args, **kwargs)