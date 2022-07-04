import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, padding_idx: int, max_len: int = 1024):
        super().__init__()
        # Compute the positional encodings once in log space.
        self.padding_idx = padding_idx
        pe = torch.zeros(max_len, model_dim, dtype=torch.float, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pe)
        self.register_buffer("position_ids", torch.arange(max_len).unsqueeze(-1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S, B = x.shape
        positions = self.position_ids[:S, :]
        return self.pos_emb[positions]


class Embedding(nn.Module):
    """

    >>> emb = Embedding(num_embeddings=10, embedding_dim=4, pad_token_id=3, learned_positional_embeddings=True)
    >>> inp = torch.tensor([[1], [1], [3]])
    >>> embedded = emb(inp)
    >>> torch.equal(embedded[0, 0, :], embedded[1, 0, :])
    False
    >>> torch.equal(embedded[2, 0, :], torch.tensor([0, 0, 0, 0], dtype=torch.float))
    True
    >>> emb._make_positions(inp).tolist()
    [[4], [5], [3]]

    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 model_dim: int,
                 pad_token_id: int,
                 norm_embeddings: bool,
                 dropout: float,
                 learned_positional_embeddings: bool,
                 max_num_embeddings: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.model_dim = model_dim
        self.learned_positional_embeddings = learned_positional_embeddings
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.norm_embeddings = norm_embeddings
        self.dropout = dropout
        self.max_num_embeddings = max_num_embeddings

        assert self.embedding_dim <= self.model_dim, "embedding_dim cannot be greater than the model_dim"
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=self.pad_token_id)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.embedding.weight[pad_token_id], 0)

        if self.embedding_dim < self.model_dim:
            self.proj_embedding = nn.Linear(self.embedding_dim, self.model_dim)
        else:
            self.proj_embedding = nn.Identity()

        if self.learned_positional_embeddings:
            self.pos_embedding = nn.Embedding(num_embeddings=self.max_num_embeddings + self.pad_token_id + 1,
                                              embedding_dim=self.model_dim,
                                              padding_idx=self.pad_token_id)
            nn.init.normal_(self.pos_embedding.weight, mean=0, std=self.model_dim ** -0.5)
            nn.init.constant_(self.pos_embedding.weight[pad_token_id], 0)
        else:
            self.pos_embedding = PositionalEncoding(model_dim=self.model_dim,
                                                    max_len=self.max_num_embeddings,
                                                    padding_idx=self.pad_token_id)

        if self.norm_embeddings:
            self.norm = nn.LayerNorm(self.model_dim)

        self.drop = nn.Dropout(self.dropout)

    def _make_positions(self, x: torch.Tensor) -> torch.Tensor:
        mask = x.ne(self.pad_token_id).int()
        return (torch.cumsum(mask, dim=0).type_as(mask) * mask).long() + self.pad_token_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: tensor of shape [SEQ_LEN x B]
        :return: tensor of shape [SEQ_LEN x B x EMBED_DIM]
        """
        emb = self.embedding(x) * math.sqrt(self.embedding_dim)
        emb = self.proj_embedding(emb)

        if self.learned_positional_embeddings:
            positions = self._make_positions(x)
            pos_emb = self.pos_embedding(positions)
        else:
            pos_emb = self.pos_embedding(x)

        emb = emb + pos_emb

        if self.norm_embeddings:
            emb = self.norm(emb)
        return self.drop(emb)
