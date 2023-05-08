#!/usr/bin/env python3

import sys
import argparse
import logging

import torch
from gensim.models import Word2Vec

from composer.conll_corpus import CONLLCorpus
from composer.model import Composer, DependencyEncoding
from composer.data import ComposerCONLLIterableDataset
from composer.utils import configure_logging, find_nps_in_tree, get_conll_file_paths

log = logging.getLogger(__name__)
configure_logging()

def main(corpus_path: str,
        train_path: str,
        token_embedding_dim: int,
        seq_len: int,
        epochs: int) :
    corpus = CONLLCorpus(corpus_path)
    word_embeddings = Word2Vec(list(corpus.get_texts()),
            vector_size = token_embedding_dim)

    composer = Composer(token_embedding_dim,
            6000,
            500,
            20,
            500,
            seq_len,
            3,
            dtype = torch.float)
    composer.token_embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.Tensor(word_embeddings.wv.vectors))

    dataset = ComposerCONLLIterableDataset([train_path],
            word_embeddings.wv.get_index,
            word_embeddings.wv.has_index_for)
    dataloader = torch.utils.data.DataLoader(dataset)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(composer.parameters(), lr = 0.001)

    for epoch in range(epochs) :
        for batch in dataloader :
            input_ids, dep_ids, head_idcs = composer.pad_inputs(*batch[:3])
            label_ids = batch[3]

            deps = DependencyEncoding(dep_ids, head_idcs)

            targets = composer.token_embedding_layer(label_ids)
            targets = targets.expand((seq_len, -1, -1)).transpose(0, 1).clone()
            target_mask = (head_idcs == 0).expand(
                    (token_embedding_dim, *head_idcs.shape)).permute((1, 2, 0))
            targets *= target_mask

            preds = composer(input_ids, deps)

            loss = loss_fn(preds, targets)
            print(f"batch loss %f" % loss.item())
            loss.backward()
            optimizer.step()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path",
            help = "path to directory with CONLL files for training",
            required = True)
    parser.add_argument("--train-path",
            help = "path to the file with a list of phrases to approximate",
            required = True)
    parser.add_argument("--epochs",
            help = "number of epochs of training to run",
            type = int,
            default = 1)
    parser.add_argument("--token-embedding-dim",
            help = "embedding dimensionality for tokens and output",
            type = int,
            default = 500)
    parser.add_argument("--seq-len",
            help = "maximum input sequence length",
            type = int,
            default = 6)
    args = parser.parse_args(sys.argv[1:])
    main(**vars(args))
