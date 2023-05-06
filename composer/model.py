from dataclasses import dataclass

import torch
import numpy as np

def generate_mask(dep_heads: torch.LongTensor,
        embedding_dim: int) :
    # populate rank-3 mask of tokens to whether each other token is their child
    # [batch_sz, seq_len, seq_len]
    dep_mask = np.zeros((*dep_heads.shape, dep_heads.shape[-1]), dtype = bool)
    for i, sample in enumerate(dep_heads) :
        for j, head in enumerate(sample) :
            dep_mask[i, head, j] = True

    # copy across embedding axis to allow multiplication
    dep_mask = np.tile(
            np.expand_dims(dep_mask, len(dep_mask.shape)),
            embedding_dim)

    return torch.Tensor(dep_mask)

@dataclass
class DependencyEncoding :
    types: torch.LongTensor
    heads: torch.LongTensor

class CompositionBlock(torch.nn.Module) :
    def __init__(self,
            token_embedding_dim: int,
            dep_embedding_dim: int,
            dep_transformed_dim: int,
            seq_len: int,
            dtype: torch.dtype = torch.float16) :
        super().__init__()
        self.seq_len = seq_len
        
        self.dep_transform = torch.nn.Bilinear(token_embedding_dim,
                dep_embedding_dim,
                dep_transformed_dim,
                dtype = dtype)
        self.dep_activation = torch.nn.Tanh()

        self.composition_transform = torch.nn.Bilinear(token_embedding_dim,
                dep_transformed_dim,
                token_embedding_dim,
                dtype = dtype)
        self.composition_activation = torch.nn.Tanh()

        self.reduction_transform = torch.nn.Linear(self.seq_len, 1)
        self.reduction_activation = torch.nn.Tanh()

    def forward(self,
            token_embeddings: torch.HalfTensor,
            dep_embeddings: torch.HalfTensor,
            dep_heads: torch.LongTensor) :

        # assert that we have an appropriate input shape (seq_len is respected)
        assert(token_embeddings.shape[1] == self.seq_len)
        assert(dep_embeddings.shape[1] == self.seq_len)
        assert(dep_heads.shape[1] == self.seq_len)

        # transform each embedding according to its dependency type relative to its head
        token_dep_embeddings = self.dep_transform(token_embeddings, dep_embeddings)
        token_dep_embeddings = self.dep_activation(token_embeddings)

        # broadcast dependency-transformed embedding
        # [batch_sz, seq_len, embedding_dim] -> [batch_sz, seq_len, seq_len, embedding_dim]
        # we want to relate (optionally) each token to each other token within a sample
        # this tensor is cloned so that the masking can be performed (else repeats are not stored separately)
        token_dep_embeddings = token_dep_embeddings.expand(self.seq_len,
                *token_dep_embeddings.shape).transpose(0, 1).clone()
        
        # create dependency mask so that each token is only going to be influenced by its children
        token_dep_mask = generate_mask(dep_heads, token_embeddings.shape[-1])
        token_dep_embeddings *= token_dep_mask

        # broadcast unchanged token embeddings for shape compatibility
        token_embeddings = token_embeddings.expand(self.seq_len,
                *token_embeddings.shape).transpose(0, 1)

        # compose each token with each of its children
        # at the end we reshape to prep for next step
        # [batch_sz, seq_len, seq_len, embedding_dim]
        # -> [batch_sz, seq_len, embedding_dim, seq_len]
        token_embeddings = self.composition_transform(token_embeddings,
                token_dep_embeddings)
        token_embeddings = self.composition_activation(token_embeddings)
        token_embeddings = token_embeddings.transpose(2, 3)

        # transform/pool down transforms of all children into single new embedding for each token
        # [batch_sz, seq_len, embedding_dim, seq_len]
        # -> [batch_sz, seq_len, embedding_dim, 1]
        # -> [batch_sz, seq_len, embedding_dim]
        token_embeddings = self.reduction_transform(token_embeddings)
        token_embeddings = token_embeddings.squeeze(3)

        return token_embeddings

class Composer(torch.nn.Module) :
    def __init__(self,
            token_embedding_dim: int,
            token_embedding_ct: int,
            dep_embedding_dim: int,
            dep_embedding_ct: int,
            dep_transformed_dim: int,
            seq_len: int,
            depth: int,
            dtype: torch.dtype = torch.float16) :
        super().__init__()
        self.depth = depth

        self.token_embedding_layer = torch.nn.Embedding(token_embedding_ct,
                token_embedding_dim,
                dtype = dtype)
        self.dep_embedding_layer = torch.nn.Embedding(dep_embedding_ct,
                dep_embedding_dim,
                dtype = dtype)

        self.composition_block = CompositionBlock(token_embedding_dim,
                dep_embedding_dim,
                dep_transformed_dim,
                seq_len,
                dtype = dtype)

    def forward(self,
            tokens: torch.LongTensor,
            deps: DependencyEncoding) :
        token_embeddings = self.token_embedding_layer(tokens)
        dep_embeddings = self.dep_embedding_layer(deps.types)

        for _ in range(self.depth) :
            token_embeddings = self.composition_block(token_embeddings,
                    dep_embeddings,
                    deps.heads)

        output_mask = (deps.heads == 0).expand(
                (token_embeddings.shape[-1], *deps.heads.shape)).permute((1, 2, 0))
        token_embeddings *= output_mask

        return token_embeddings
