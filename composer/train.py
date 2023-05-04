#!/usr/bin/env python3

import sys
import argparse

import torch
from gensim.models import Word2Vec

from conll_corpus import CONLLCorpus
from model import Composer

def main(corpus_path) :
    corpus = CONLLCorpus(corpus_path)
    word_embeddings = Word2Vec(list(corpus.get_texts()))

    composer = Composer(500, 6000, 500, 20, 500, 4, 3, dtype = torch.float)
    composer.token_embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.Tensor(word_embeddings.wv.vectors))

    print(composer)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path",
            help = "path to directory with CONLL files for training",
            required = True)
    args = parser.parse_args(sys.argv[1:])
    main(**vars(args))
