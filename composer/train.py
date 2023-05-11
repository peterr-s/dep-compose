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
        val_path: str,
        token_embedding_dim: int,
        seq_len: int,
        epochs: int,
        batch_sz: int,
        val_batch_sz: int,
        dep_type_ct: int) :
    log.info("training embedding model")
    corpus = CONLLCorpus(corpus_path)
    word_embeddings = Word2Vec(list(corpus.get_texts()),
            vector_size = token_embedding_dim)

    log.info("setting up composer")
    composer = Composer(token_embedding_dim,
            15000,
            2000,
            dep_type_ct,
            token_embedding_dim,
            seq_len,
            3,
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

    loss_fn = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(composer.parameters(), lr = 0.001)

    log.info("loading composer training data")
    dataset = ComposerCONLLIterableDataset(get_conll_file_paths(train_path),
            word_embeddings.wv.get_index,
            word_embeddings.wv.has_index_for,
            composer.pad_inputs)
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size = batch_sz)
    
    dataset_val = ComposerCONLLIterableDataset(get_conll_file_paths(val_path),
            word_embeddings.wv.get_index,
            word_embeddings.wv.has_index_for,
            composer.pad_inputs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
            batch_size = val_batch_sz)

    log.info(f"starting training run, {epochs=}")
    for epoch_no in range(epochs) :
        loss_total = 0
        batch_ct = 0
        iterator_val = iter(dataloader_val)
        for batch_no, batch in enumerate(dataloader) :
            input_ids, dep_ids, head_idcs, label_ids = batch
            label_ids = label_ids.squeeze(1)

            input_ids = input_ids.to(device)
            dep_ids = dep_ids.to(device)
            head_idcs = head_idcs.to(device)
            label_ids = label_ids.to(device)

            deps = DependencyEncoding(dep_ids, head_idcs)
            targets = composer.token_embedding_layer(label_ids)

            preds = composer(input_ids, deps)
            preds = torch.sum(preds, dim = 1)

            loss = loss_fn(preds, targets, torch.ones(batch_sz, device = device))
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            if not batch_no % 20 :
                batch_val = next(iterator_val, None)
                if not batch_val :
                    iterator_val = iter(dataloader_val)
                    batch_val = next(iterator_val)

                input_ids, dep_ids, head_idcs, label_ids = batch_val
                label_ids = label_ids.squeeze(1)

                input_ids = input_ids.to(device)
                dep_ids = dep_ids.to(device)
                head_idcs = head_idcs.to(device)
                label_ids = label_ids.to(device)

                deps = DependencyEncoding(dep_ids, head_idcs)
                targets = composer.token_embedding_layer(label_ids)

                preds = composer(input_ids, deps)
                preds = torch.sum(preds, dim = 1)

                loss_val = loss_fn(preds, targets, torch.ones(batch_sz, device = device)).item()

                loss_avg = loss_total / min(batch_no + 1, 20)
                log.info(f"{epoch_no=}\t{batch_no=}\t{loss_avg=}\t{loss_val=}")
                loss_total = 0

            batch_ct = batch_no

        loss_avg = loss_total / ((batch_ct % 20) + 1)
        log.info(f"{epoch_no=} final {loss_avg=}")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path",
            help = "path to directory with CONLL files for embedding training",
            required = True)
    parser.add_argument("--train-path",
            help = "path to directory with a list of phrases to approximate",
            required = True)
    parser.add_argument("--val-path",
            help = "path to directory with a list of phrases to validate on",
            required = True)
    parser.add_argument("--epochs",
            help = "number of epochs of training to run",
            type = int,
            default = 1)
    parser.add_argument("--batch-sz",
            help = "batch size to use in training",
            type = int,
            default = 5)
    parser.add_argument("--val-batch-sz",
            help = "batch size to use for validation",
            type = int,
            default = 20)
    parser.add_argument("--token-embedding-dim",
            help = "embedding dimensionality for tokens and output",
            type = int,
            default = 300)
    parser.add_argument("--dep-type-ct",
            help = "number of dependency types for which to train transforms",
            type = int,
            default = 52)
    parser.add_argument("--seq-len",
            help = "maximum input sequence length",
            type = int,
            default = 6)
    args = parser.parse_args(sys.argv[1:])
    main(**vars(args))
