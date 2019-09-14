import torch.distributed as dist

def reduce_gradients():
    pass


def average_reduce():
    pass


def get_rank():
    return dist.get_rank()


def get_word_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_rank()


class DistModule():
    pass