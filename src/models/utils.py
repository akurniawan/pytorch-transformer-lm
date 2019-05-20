import torch


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    elif isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(_h) for _h in h)