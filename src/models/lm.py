import torch
import torch.nn as nn


def _build_rnns(params):
    rnn_type = {"lstm": nn.LSTM, "gru": nn.GRU}

    rnns = []
    for i in range(params.num_layers):
        if i == 0:
            input_size = params.input_size
        else:
            input_size = params.hidden_size * \
                (2 if params.bidirectional else 1)

        if i == params.num_layers - 1:
            if params.weight_tying:
                hidden_size = params.input_size
            else:
                hidden_size = params.hidden_size
        else:
            hidden_size = params.hidden_size

        rnn = rnn_type[params.rnn_type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=params.bidirectional)
        rnns.append(rnn)

    return rnns


class RNNLanguageModel(nn.Module):
    def __init__(self, hyperparams, encoder):
        super(RNNLanguageModel, self).__init__()

        self.rnns = nn.ModuleList(_build_rnns(hyperparams))
        self.encoder = encoder

        self._hyperparams = hyperparams

    def forward(self, X, hidden):
        if not isinstance(X, torch.Tensor):
            raise TypeError("input must be an instance of torch.Tensor!")
        if not isinstance(hidden, tuple):
            raise TypeError("initial hidden units must be "
                            "an instance of tuple!")

        outputs = self.encoder(X)
        hidden_n = []

        for layer_n, rnn in enumerate(self.rnns):
            outputs, this_hidden = rnn(outputs, hidden[layer_n])
            hidden_n.append(this_hidden)

        return outputs, hidden_n

    def init_hidden_layer(self, batch_size):
        init_hidden = []
        for rnn in self.rnns:
            hidden_size = rnn.weight_hh_l0.size(1)
            first_dim = 2 if self._hyperparams.bidirectional else 1

            param_h = torch.randn(first_dim, batch_size, hidden_size)
            param_c = torch.randn(first_dim, batch_size, hidden_size)
            init_hidden.append((param_h, param_c))

        return tuple(init_hidden)


class TransformerLanguageModel(nn.Module):
    def __init__(self, embedding, encoder, decoder=None):
        super(TransformerLanguageModel, self).__init__()

        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, words, past=None):
        emb = self.embedding(words)
        out, past = self.encoder(emb, past)
        if self.decoder:
            out = self.decoder(out)

        return out, past

    def init_weight(self):
        self.embedding.init_weight()
        self.encoder.init_weight()
        if self.decoder:
            nn.init.uniform_(self.decoder._module.weight, -0.1, 0.1)