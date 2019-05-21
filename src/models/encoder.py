import torch
import math
import torch.nn as nn

from ..modules.attention import MultiHeadAttention
from ..modules.ffn import PositionWiseFFN

from typing import NamedTuple, List


class TransformerEncoder(nn.Module):
    class Hyperparams(NamedTuple):
        """
        Args:
            query_dim: Number of words in dictionary
            att_num_units: Number of embedding dimension
            ffn_num_unit: Dropout probability for embedding to
                hidden units layer
            max_ext: Dropout probability for embedding matrix
        """
        query_dim: int = 512
        att_num_units: List[int] = [512, 512, 512]
        ffn_num_unit: int = 2048
        max_ext: float = math.inf

    def __init__(self, params):
        super(TransformerEncoder, self).__init__()

        self.encoder = self._build_model(
            params.query_dim, params.att_num_units, params.ffn_num_unit)
        self._max_ext = params.max_ext

    def _build_model(self, query_dim, att_num_units, ffn_num_unit):
        layers = []

        for unit in att_num_units:
            layer = nn.ModuleList([
                MultiHeadAttention(query_dim, query_dim, unit, is_masked=True),
                PositionWiseFFN(unit, ffn_num_unit)
            ])
            layers.append(layer)

        return nn.ModuleList(layers)

    def forward(self, query, past=None):
        out = None
        curr_past = []
        for layer_id, dec in enumerate(self.encoder):
            this_past = None if past is None else past[layer_id]
            keys = self._update_keys(this_past, query)
            res1 = dec[0](query, keys)
            res2 = dec[1](res1)
            query = res2
            out = res2

            new_past = self._update_past(this_past, out)
            curr_past.append(new_past)

        return out, curr_past

    def _update_keys(self, past, query):
        if past is not None:
            keys = torch.cat([past, query], dim=1)
        else:
            keys = query
        return keys

    def _update_past(self, past, out):
        with torch.no_grad():
            if past is not None:
                keys = torch.cat([past, out], dim=1)
                # truncate the length
                seq_len = keys.size(1)
                if self._max_ext - seq_len < 0:
                    trunc_start_idx = abs(self._max_ext - seq_len) - 1
                else:
                    trunc_start_idx = 0
                keys = keys[:, trunc_start_idx:, :]
            else:
                keys = out
            return keys.detach()

    def init_weight(self):
        for layer in self.encoder:
            layer[0].init_weight()
            layer[1].init_weight()