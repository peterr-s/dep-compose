#!/usr/bin/env python

import dynet
import pyconll
from gensim.models import KeyedVectors

import numpy as np

import sys
from config import DefaultConfig

# usage: model.py train test embeddings
if len(sys.argv) < 4 :
    print("usage: %s train test embeddings" % sys.argv[0], file = sys.stderr)
    exit(1)

# load config from file
config = DefaultConfig()

# load embedding model
embeddings = KeyedVectors.load_word2vec_format(sys.argv[3], binary = True)
word_dim = embeddings.vector_size
null_word = np.zeros(word_dim)

# load parses
parse_train = pyconll.load_from_file(sys.argv[1])
parse_test = pyconll.load_from_file(sys.argv[2])

# define model parameters
model = dynet.ParameterCollection()

# define a layer for each possible dependency

# define graph building operation
def generate_graph(parse) :

# run training
for parse in parse_train :
    loss = generate_graph(parse)

# run eval
