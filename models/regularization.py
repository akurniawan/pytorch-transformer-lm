import torch.nn as nn
import torch.nn.functional as F


class DropEmbedding(nn.Module):
    def __init__(self, embedding, params):
        super(DropEmbedding, self).__init__()

        self.embedding = embedding
        self.dropout = params.dropoute
        self.scale = params.scale_embedding
        self.vocab_size = params.vocab_size

        # Embedding parameters
        self.max_norm = self.embedding.max_norm
        self.norm_type = self.embedding.norm_type
        self.scale_grad_by_freq = self.embedding.scale_grad_by_freq
        self.sparse = self.embedding.sparse

    def forward(self, X):
        if self.dropout:
            # Create new tensor from previously created embedding.
            # This to make us easier to grab sub parts of the embedding
            # without creating new tensor
            mask = self.embedding.weight.data.new()
            # Resize by only creating a single tensor consist of
            # random weight thahth later will be expanded to the original
            # tensor size
            mask = mask.resize_((self.vocab_size, 1))
            # Draw bernoulli distriboution based on the dropout probability
            mask = mask.bernoulli_(1 - self.dropout)
            # Expand the masking that previously drawn from bernoulli
            # distribution to match the original size of the tensor
            mask = mask.expand_as(self.embedding.weight) / (1 - self.dropout)

            masked_embed_weight = mask * self.embedding.weight
        else:
            masked_embed_weight = self.embedding.weight
        if self.scale:
            masked_embed_weight = self.scale.expand_as(
                masked_embed_weight) * masked_embed_weight

        padding_idx = self.embedding.padding_idx
        if padding_idx is None:
            padding_idx = -1

        # Create new functional embedding with the same parameter as the
        # previous one, the only difference is on the weight
        X = F.embedding(X, masked_embed_weight, padding_idx, self.max_norm,
                        self.norm_type, self.scale_grad_by_freq, self.sparse)
        return X


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
