import torch.nn as nn

from models.regularization import DropEmbedding


class RNNEncoder(nn.Module):
    def __init__(self, params):
        super(RNNEncoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=params.vocab_size,
            embedding_dim=params.input_size,
            padding_idx=params.padding_idx)
        self.drop_embedding = DropEmbedding(
            embedding=self.embedding, params=params)

    def forward(self, X):
        return self.drop_embedding(X)
