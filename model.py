from dataclasses import dataclass

import torch

@dataclass
class DependencyEncoding :
    types: torch.LongTensor
    heads: torch.LongTensor

class CompositionBlock(torch.nn.Module) :
    def __init__(self,
            token_embedding_dim: int,
            dep_embedding_dim: int,
            dep_transformed_dim: int,
            dtype: torch.dtype = torch.float16) :
        super().__init__()
        
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

    def forward(self,
            token_embeddings: torch.HalfTensor,
            dep_embeddings: torch.HalfTensor,
            dep_heads: torch.LongTensor) :
        token_dep_embeddings = self.dep_transform(token_embeddings, dep_embeddings)
        token_dep_embeddings = self.dep_activation(token_embeddings)

        token_embeddings = self.composition_transform(token_embeddings,
                token_dep_embeddings)
        token_embeddings = self.composition_activation(token_embeddings)

        return token_embeddings

class Composer(torch.nn.Module) :
    def __init__(self,
            token_embedding_dim: int,
            token_embedding_ct: int,
            dep_embedding_dim: int,
            dep_embedding_ct: int,
            dep_transformed_dim: int,
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

        return token_embeddings
