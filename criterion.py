import torch.nn as nn

_MEDIUM_TOKENS = 65000
_HIGH_TOKENS = 500000


def lm_criterion(params):
    if params.weight_tying:
        in_features = 2 * params.input_size \
            if params.bidirectional else params.hidden_size
    else:
        in_features = 2 * params.hidden_size \
            if params.bidirectional else params.hidden_size

    splits = []
    if params.vocab_size > _MEDIUM_TOKENS:
        splits = [2800, 20000, 760000]
    elif params.vocab_size > _HIGH_TOKENS:
        splits = [4200, 35000, 180000]
    splits += [params.vocab_size - 2]

    criterion = nn.AdaptiveLogSoftmaxWithLoss(
        in_features=in_features, n_classes=params.vocab_size, cutoffs=splits)

    return criterion
