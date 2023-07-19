from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import numpy as np

def generate_mask(dep_heads: torch.LongTensor,
        embedding_dim: int) -> torch.BoolTensor :
    # populate rank-3 mask of tokens to whether each other token is their child
    # [batch_sz, seq_len, seq_len]
    dep_mask = np.zeros((*dep_heads.shape, dep_heads.shape[-1]), dtype = bool)
    for i, sample in enumerate(dep_heads) :
        for j, head in enumerate(sample) :
            if head < dep_heads.shape[-1] :
                dep_mask[i, head, j] = True

    # copy across embedding axis to allow multiplication
    dep_mask = np.tile(
            np.expand_dims(dep_mask, len(dep_mask.shape)),
            embedding_dim)

    return torch.BoolTensor(dep_mask)

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
        self.dtype = dtype
        
        self.dep_transform = torch.nn.Bilinear(token_embedding_dim,
                dep_embedding_dim,
                dep_transformed_dim,
                dtype = self.dtype)
        self.dep_activation = torch.nn.Tanh()

        self.composition_transform = torch.nn.Bilinear(token_embedding_dim,
                dep_transformed_dim,
                token_embedding_dim,
                dtype = self.dtype)
        self.composition_activation = torch.nn.Tanh()

        self.reduction_transform = torch.nn.Linear(self.seq_len, 1, dtype = self.dtype)
        self.reduction_activation = torch.nn.Softmax(dim = 1)

    def forward(self,
            token_embeddings: torch.Tensor,
            dep_embeddings: torch.Tensor,
            dep_heads: torch.LongTensor) :

        # assert that we have an appropriate input shape (seq_len is respected)
        assert(token_embeddings.shape[1] == self.seq_len)
        assert(dep_embeddings.shape[1] == self.seq_len)
        assert(dep_heads.shape[1] == self.seq_len)

        # cast embeddings to correct dtype
        if token_embeddings.dtype != self.dtype :
            token_embeddings = token_embeddings.to(self.dtype)
        if dep_embeddings.dtype != self.dtype :
            dep_embeddings = dep_embeddings.to(self.dtype)

        # transform each embedding according to its dependency type relative to its head
        token_dep_embeddings = self.dep_transform(token_embeddings, dep_embeddings)

        # broadcast dependency-transformed embedding
        # [batch_sz, seq_len, embedding_dim] -> [batch_sz, seq_len, seq_len, embedding_dim]
        # we want to relate (optionally) each token to each other token within a sample
        # this tensor is cloned so that the masking can be performed (else repeats are not stored separately)
        token_dep_embeddings = token_dep_embeddings.expand(self.seq_len,
                *token_dep_embeddings.shape).transpose(0, 1).clone()
        
        # create dependency mask so that each token is only going to be influenced by its children
        token_dep_mask = generate_mask(dep_heads, token_embeddings.shape[-1]).to(self.device())
        token_dep_embeddings = token_dep_embeddings.where(token_dep_mask,
                torch.zeros(token_dep_embeddings.shape, device = self.device(), dtype = self.dtype))
        token_dep_embeddings = self.dep_activation(token_dep_embeddings)

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

    def device(self) :
        return next(self.parameters()).device

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
                (token_embeddings.shape[-1], *deps.heads.shape)).permute((1, 2, 0)).to(self.device())
        token_embeddings *= output_mask

        return token_embeddings

    def device(self) -> torch.device :
        return next(self.parameters()).device

    def truncate_input(self, tensor: torch.LongTensor) -> torch.LongTensor :
        return tensor[:self.composition_block.seq_len].clone()

    def truncate_inputs(self,
            tokens: Optional[torch.LongTensor] = None,
            deps: Optional[torch.LongTensor] = None,
            heads: Optional[torch.LongTensor] = None) -> Tuple[Optional[torch.LongTensor]] :
        return (self.truncate_input(tokens) if tokens is not None else None,
                self.truncate_input(deps) if deps is not None else None,
                self.truncate_input(heads) if heads is not None else None)

    def pad_input(self, tensor: torch.LongTensor) -> torch.LongTensor :
        return torch.cat((tensor,
                        torch.zeros(self.composition_block.seq_len - tensor.shape[0],
                            dtype = torch.long)))

    def pad_inputs(self,
            tokens: Optional[torch.LongTensor] = None,
            deps: Optional[torch.LongTensor] = None,
            heads: Optional[torch.LongTensor] = None) -> Tuple[Optional[torch.LongTensor]] :
        tokens, deps, heads = self.truncate_inputs(tokens, deps, heads)
        return (self.pad_input(tokens) if tokens is not None else None,
                self.pad_input(deps) if deps is not None else None,
                self.pad_input(heads) if heads is not None else None)

class DynamicComposer(Composer) :
    def forward(self,
            tokens: torch.LongTensor,
            layered_deps: List[DependencyEncoding]) :
        token_embeddings = self.token_embedding_layer(tokens)

        for deps in layered_deps :
            dep_embeddings = self.dep_embedding_layer(deps.types)

            token_embeddings = self.composition_block(token_embeddings,
                    dep_embeddings,
                    deps.heads)

        output_mask = (layered_deps[-1].heads == 0).expand(
                (token_embeddings.shape[-1], *layered_deps[-1].heads.shape)).permute((1, 2, 0)).to(self.device())
        token_embeddings *= output_mask

        return token_embeddings

