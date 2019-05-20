class RNNLanguageModelParams(object):
    def __init__(self,
                 rnn_type="lstm",
                 num_layers=3,
                 hidden_size=256,
                 bidirectional=True,
                 weight_tying=True,
                 input_size=300,
                 vocab_size=10,
                 padding_idx=0,
                 dropoute=0.1,
                 scale_embedding=None):
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.weight_tying = weight_tying
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx

        self.dropoute = dropoute
        self.scale_embedding = scale_embedding
