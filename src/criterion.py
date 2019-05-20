import torch.nn as nn

_MEDIUM_TOKENS = 65000
_HIGH_TOKENS = 500000


def lm_criterion(in_features, vocab_size):
    # if weight_tying:
    #     in_features = 2 * input_size \
    #         if bidirectional else hidden_size
    # else:
    #     in_features = 2 * hidden_size \
    #         if bidirectional else hidden_size

    splits = []
    if vocab_size > _MEDIUM_TOKENS:
        splits = [2800, 20000, 760000]
    elif vocab_size > _HIGH_TOKENS:
        splits = [4200, 35000, 180000]
    splits += [vocab_size - 2]

    criterion = nn.AdaptiveLogSoftmaxWithLoss(
        in_features=in_features, n_classes=vocab_size, cutoffs=splits)

    return criterion
