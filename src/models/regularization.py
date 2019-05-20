import torch
import torch.nn as nn
import torch.nn.functional as F


class LockedDropout(nn.Module):
    """Following https://arxiv.org/pdf/1512.05287.pdf to create
    locked dropout. The way it works is basically locking the mask
    matrix for every steps.

    Args:
        dropout (float): Dropout rate
    """

    def __init__(self, dropout=0.5):
        super(LockedDropout, self).__init__()

        self.dropout_rate = dropout

    def forward(self, x):
        if self.training or self.dropout_rate != 0.:
            m = torch.empty(
                1, x.size(1), x.size(2), device=x.device,
                dtype=x.dtype).bernoulli_(1 - self.dropout_rate)
            mask = m / (1 - self.dropout_rate)
            mask = mask.expand_as(x)
            x = x * mask
        return x


class WeightDrop(nn.Module):
    def __init__(self, module, weights, dropout_rate=0, variational=False):
        """Dropout class for RNN based neural networ

        Args:
            module (nn.Module): lstm based module
            weights (List): List of weight name that will
                be droppedout
            dropout_rate (int, optional): Defaults to 0.
                The rate of the dropout
            variational (bool, optional): Defaults to False.
                Flag to turn on variational dropout
        """
        super(WeightDrop, self).__init__()

        self.module = module
        self.weights = weights
        self.dropout_rate = dropout_rate
        self.variational = variational

        self._raw_weights = {}
        self._setup()

    def _setup(self):
        """Setting up raw weights from RNNBase module
        to this class. The value later will be used in forward
        """
        for w_name in self.weights:
            raw_w = getattr(self.module, w_name)
            self.register_parameter(w_name + "_raw", raw_w)

    def forward(self, *args):
        """Run RNNBase forward, but before that, will update the
        original weights of RNNBase by applying dropout

        Returns:
            Return the value of RNNBase
        """
        for w_name in self.weights:
            raw_w = getattr(self, w_name + "_raw")
            w = F.dropout(raw_w, p=self.dropout_rate, training=self.training)
            del self.module._parameters[w_name]
            self.module._parameters[w_name] = w

        return self.module(*args)
