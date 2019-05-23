import torch
import math
import torch.nn as nn


class Gelu(nn.Module):
    def __init__(self):
        super().__init__()
        self._c = math.sqrt(math.pi / 2)

    def forward(self, x):
        divisor = torch.tanh(self._c * (x + 0.044715 * x**3))
        return 0.5 * x * (1 + divisor)


class PositionWiseFFN(nn.Module):
    def __init__(self, feature_size, num_units=2048, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self._dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(feature_size, num_units), Gelu(),
            nn.Dropout(self._dropout), nn.Linear(num_units, feature_size),
            nn.Dropout(self._dropout))
        self.ln = nn.LayerNorm(feature_size)

    def forward(self, X):
        ffn = self.ffn(X)
        # residual network
        ffn += X
        ffn = self.ln(ffn)

        return ffn

    def init_weight(self):
        for idx in range(len(self.ffn)):
            if hasattr(self.ffn[idx], "weight"):
                nn.init.uniform_(self.ffn[idx].weight, -0.1, 0.1)