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
        epochs: int,
        batch_sz: int) :
    log.info("training embedding model")
    corpus = CONLLCorpus(corpus_path)
    word_embeddings = Word2Vec(list(corpus.get_texts()),
            vector_size = token_embedding_dim)

    log.info("setting up composer")
    composer = Composer(token_embedding_dim,
            15000,
            300,
            20,
            token_embedding_dim,
            seq_len,
            2,
            dtype = torch.float)
    log.debug(f"set up {composer=}")
    composer.token_embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.Tensor(word_embeddings.wv.vectors))
    log.debug(f"replaced embedding layer; {composer=}")
    log.info(f"model has {sum(p.numel() for p in composer.parameters())} params")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    composer.to(device)
    composer.token_embedding_layer.to(device)
    log.info(f"moved model to {device}")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(composer.parameters(), lr = 0.001)

    log.info("loading composer training data")
    dataset = ComposerCONLLIterableDataset([train_path],
            word_embeddings.wv.get_index,
            word_embeddings.wv.has_index_for,
            composer.pad_inputs)
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size = batch_sz)

    log.info(f"starting training run, {epochs=}")
    for epoch_no in range(epochs) :
        loss_total = 0
        batch_ct = 0
        for batch_no, batch in enumerate(dataloader) :
            input_ids, dep_ids, head_idcs, label_ids = batch
            label_ids = label_ids.squeeze(1)
            input_ids = input_ids.to(device)
            dep_ids = dep_ids.to(device)
            head_idcs = head_idcs.to(device)
            label_ids = label_ids.to(device)

            deps = DependencyEncoding(dep_ids, head_idcs)

            targets = composer.token_embedding_layer(label_ids)
            targets = targets.expand((seq_len, -1, -1)).transpose(0, 1).clone()
            target_mask = (head_idcs == 0).expand(
                    (token_embedding_dim, *head_idcs.shape)).permute((1, 2, 0))
            targets *= target_mask

            preds = composer(input_ids, deps)

            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            if not batch_no % 20 :
                loss_avg = loss_total / (batch_no + 1)
                log.info(f"{epoch_no=} {batch_no=} {loss_avg=}")
            batch_ct = batch_no

        loss_avg = loss_total / batch_ct
        log.info(f"{epoch_no=} final {loss_avg=}")

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
    parser.add_argument("--batch-sz",
            help = "batch size to use in training",
            type = int,
            default = 5)
    parser.add_argument("--token-embedding-dim",
            help = "embedding dimensionality for tokens and output",
            type = int,
            default = 300)
    parser.add_argument("--seq-len",
            help = "maximum input sequence length",
            type = int,
            default = 6)
    args = parser.parse_args(sys.argv[1:])
    main(**vars(args))
