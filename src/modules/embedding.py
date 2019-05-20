import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

from typing import NamedTuple
from ..models.regularization import LockedDropout


class DropEmbedding(nn.Module):
    class Hyperparams(NamedTuple):
        """Dropped Embedding Hyperparameters
        Args:
            ntokens: Number of words in dictionary
            ninp: Number of embedding dimension
            dropouti: Dropout probability for embedding to hidden units layer
            dropoute: Dropout probability for embedding matrix
        """
        ntokens: int
        ninp: int = 400
        padding_idx: int = 0
        dropouti: float = 0.65
        dropoute: float = 0.1
        scale: float = 0.0

    def __init__(self, params: Hyperparams):
        super(DropEmbedding, self).__init__()

        self.embedding = nn.Embedding(
            params.ntokens, params.ninp, padding_idx=params.padding_idx)
        self.dropout = params.dropoute
        self.scale = torch.tensor(params.scale)
        self.vocab_size = params.ntokens

        # Embedding parameters
        self.max_norm = self.embedding.max_norm
        self.norm_type = self.embedding.norm_type
        self.scale_grad_by_freq = self.embedding.scale_grad_by_freq
        self.sparse = self.embedding.sparse

        self.lock_dropout = LockedDropout(params.dropouti)

    def forward(self, X):
        if self.dropout:
            # Create new tensor from previously created embedding.
            # This to make us easier to grab sub parts of the embedding
            # without creating new tensor
            mask = self.embedding.weight.data.new()
            # Resize by only creating a single tensor consist of
            # random weight that later will be expanded to the original
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
        X = self.lock_dropout(X)
        return X

    def init_weight(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)


class WordCharCNNEmbedding(nn.Module):
    """The character embedding is built upon CNN and pooling layer
    with dropout applied before the convolution and after the pooling.
    """

    class Hyperparams(NamedTuple):
        """Char CNN Hyperparameters
        Args:
            ntokens: Number of characters in vocabulary
            char_embedding_dim: The number of embedding of the character
            char_padding_idx: The index of padding token
            dropout: Dropout rate for the connection between Char
                embedding and Conv layer
            kernel_size: The size of the convolutional layer kernel
            out_channels: Number of final embedding of the characters
        """
        ntokens: int
        char_embedding_dim: int = 30
        char_padding_idx: int = 1
        dropout: float = 0.5
        kernel_size: int = 3
        out_channels: int = 30

    def __init__(self, char_cnn_params: Hyperparams):
        super(WordCharCNNEmbedding, self).__init__()
        self._params = char_cnn_params

        self.char_embedding = nn.Embedding(char_cnn_params.ntokens,
                                           char_cnn_params.char_embedding_dim,
                                           char_cnn_params.char_padding_idx)
        self.conv_embedding = nn.Sequential(
            nn.Dropout(p=char_cnn_params.dropout),
            nn.Conv1d(
                in_channels=char_cnn_params.char_embedding_dim,
                out_channels=char_cnn_params.out_channels,
                kernel_size=char_cnn_params.kernel_size,
                padding=char_cnn_params.kernel_size - 1),
            nn.AdaptiveMaxPool1d(1))
        self.out_dropout = nn.Dropout(p=char_cnn_params.dropout)

    def init_weights(self):
        """Initialize the weight of character embedding with xavier
        and reinitalize the padding vectors to zero
        """

        self.char_embedding.weight.data.uniform_(-0.1, 0.1)
        # Reinitialize vectors at padding_idx to have 0 value
        self.char_embedding.weight.data[
            self._params.char_padding_idx].uniform_(0, 0)

    def forward(self, chars):
        """Run the forward calculation of the char-cnn embedding
        model.

        Args:
            chars (torch.Tensor): An integer tensor with the size of
                [seq_len x batch x char_size]

        Returns:
            char_embedding_vec (torch.Tensor): An embedding tensor with
                the size of [batch x seq_len x out_channels]
        """
        char_embedding_vec = self.char_embedding(chars)
        # Reshape the character embedding to the size of
        # [batch * seq_len, char_len, char_dim]
        char_embedding_vec = char_embedding_vec.view(
            -1, char_embedding_vec.size(2),
            char_embedding_vec.size(3)).contiguous()
        # Transpose the embedding into [batch * seq_len, char_dim, char_len]
        char_embedding_vec = char_embedding_vec.transpose(1, 2).contiguous()
        # Apply char embedding with dropout and convolution
        # layers so the dim now will be [batch * seq_len, out_channel, new_len]
        char_embedding_vec = self.conv_embedding(char_embedding_vec)
        char_embedding_vec = char_embedding_vec.squeeze(-1)
        # Revert the size back to [seq_len, batch, out_channel]
        char_embedding_vec = char_embedding_vec.view(
            chars.size(0), chars.size(1), -1).contiguous()
        char_embedding_vec = self.out_dropout(char_embedding_vec)

        return char_embedding_vec


class TransformerEmbedding(nn.Module):
    def __init__(self,
                 embedding,
                 max_length,
                 embedding_size,
                 use_positional_embedding=True):
        super(TransformerEmbedding, self).__init__()
        self._use_positional_embedding = use_positional_embedding

        self.embedding = embedding

        if use_positional_embedding:
            self.pos_embedding = nn.Embedding(max_length, embedding_size)
            pos_enc_weight = self._sin_cos_enc(max_length, embedding_size)
            self.pos_embedding.weight = pos_enc_weight

    def _sin_cos_enc(self, max_length, embedding_size):
        position_enc = np.array([[
            pos / np.power(10000, 2 * i / embedding_size)
            for i in range(embedding_size)
        ] for pos in range(max_length)],
                                dtype=np.float32)

        # put sinusodial on even position
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        # put cosine on odd position
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        return Parameter(torch.from_numpy(position_enc))

    def forward(self, X):
        word_embedding = self.embedding(X)
        if self._use_positional_embedding:
            T = X.size(1)
            pos = torch.arange(T, device=X.device).expand(X.size()).long()
            pos_embedding = self.pos_embedding(pos)
            word_embedding += pos_embedding

        return word_embedding

    def init_weight(self):
        self.embedding.init_weight()


class OneHotEmbedding(nn.Module):
    def __init__(self, num_class):
        super(OneHotEmbedding, self).__init__()
        self.embed = nn.Embedding(num_class, num_class)
        self.embed.weight.data = self._build_onehot(num_class)
        # to prevent the weight getting trained
        self.embed.weight.requires_grad = False

    def _build_onehot(self, num_class):
        onehot = torch.eye(num_class)
        return onehot

    def forward(self, x):
        return self.embed(x)
