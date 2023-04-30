import torch
import transformers

class DependencyEncoding :
    def __init__(self, types, heads) :
        self.types = types
        self.heads = heads

class CompositionBlock(torch.nn.Module) :
    def __init__(token_embedding_dim,
            dep_embedding_dim,
            dep_transformed_dim) :
        super().__init__()
        
        self.dep_transform = torch.nn.Bilinear(token_embedding_dim,
                dep_embedding_dim,
                dep_transformed_dim)
        self.composition_transform = torch.nn.Bilinear(token_embedding_dim,
                dep_transformed_dim,
                token_embedding_dim)

    def forward(self, token_embeddings, dep_embeddings, dep_heads) :
        token_dep_embeddings = self.dep_transform(token_embeddings, dep_embeddings)
        token_dep_embeddings = torch.nn.Tanh(token_embeddings)

        token_embeddings = self.composition_transform(token_embeddings,
                token_dep_embeddings)
        token_embeddings = torch.nn.Tanh(token_embeddings)

        return token_embeddings

class Composer(transformers.PreTrainedModel) :
    def __init__(token_embedding_dim,
            token_embedding_ct,
            dep_embedding_dim,
            dep_embedding_ct,
            depth) :
        super().__init__()
        self.depth = depth

        self.token_embedding_layer = torch.nn.Embedding(token_embedding_ct,
                token_embedding_dim)
        self.dep_embedding_layer = torch.nn.Embedding(dep_embedding_ct,
                dep_embedding_dim)

        self.composition_block = CompositionBlock(token_embedding_dim,
                dep_embedding_dim)

    def forward(self, tokens, deps) :
        token_embeddings = self.token_embedding_layer(tokens)
        dep_embeddings = self.dep_embedding_layer(deps.types)

        for _ in range(self.depth) :
            token_embeddings = self.composition_block(token_embeddings,
                    dep_embeddings,
                    deps.heads)

        return token_embeddings
