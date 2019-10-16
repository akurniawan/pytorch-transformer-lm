import torch
import torch.nn as nn

from ..modules.attention import MultiHeadAttention
from ..modules.ffn import PositionWiseFFN

from typing import List


class TransformerEncoder(nn.Module):
    """
    Args:
        query_dim: Number of words in dictionary
        att_num_units: Number of embedding dimension
        ffn_num_unit: Dropout probability for embedding to
            hidden units layer
        max_ext: Dropout probability for embedding matrix
    """
    def __init__(self, query_dim: int, att_num_units: List[int],
                 ffn_num_unit: int, max_ext: float):
        super(TransformerEncoder, self).__init__()

        self.encoder = self._build_model(query_dim, att_num_units,
                                         ffn_num_unit)
        self._max_ext = max_ext

    def _build_model(self, query_dim, att_num_units, ffn_num_unit):
        layers = []

        for unit in att_num_units:
            layer = nn.ModuleList([
                MultiHeadAttention(query_dim, query_dim, unit, is_masked=True),
                PositionWiseFFN(unit, ffn_num_unit)
            ])
            layers.append(layer)

        return nn.ModuleList(layers)

    def forward(self, query):
        out = None
        curr_past = []
        past = None
        for layer_id, enc in enumerate(self.encoder):
            this_past = None if past is None else past[layer_id]
            res1, model_past = enc[0](query, query, this_past)
            res2 = enc[1](res1)
            query = res2
            out = res2

            # new_past = self._update_past(this_past, model_past)
            # curr_past.append(new_past)

        return out, curr_past

    def _update_past(self, past, out):
        with torch.no_grad():
            if past is not None:
                k_past = torch.cat([past[0], out[0]], dim=1)
                v_past = torch.cat([past[1], out[1]], dim=1)
                # truncate the length
                seq_len = k_past.size(1)
                if self._max_ext - seq_len < 0:
                    trunc_start_idx = abs(self._max_ext - seq_len) - 1
                else:
                    trunc_start_idx = 0
                k_past = k_past[:, trunc_start_idx:, :]
                v_past = v_past[:, trunc_start_idx:, :]
            else:
                k_past = out[0]
                v_past = out[1]
            return (k_past.detach(), v_past.detach())

    def init_weight(self):
        for layer in self.encoder:
            layer[0].init_weight()
            layer[1].init_weight()