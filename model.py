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

# load parses
parse_train = pyconll.load_from_file(sys.argv[1])
parse_test = pyconll.load_from_file(sys.argv[2])

# define model parameters
model = dynet.ParameterCollection()

# find all possible dependency types
dep_types = set()
for parse in parse_train :
    for word in parse :
        dep_types.add(word.deprel)
for parse in parse_test :
    for word in parse :
        dep_types.add(word.deprel)

# load embedding model
embeddings = KeyedVectors.load_word2vec_format(sys.argv[3], binary = True)
word_dim = embeddings.vector_size
null_word = np.zeros(word_dim)

# define a layer for each possible dependency
dep_layers = dict()
for dep_type in dep_types :
    dep_layers[dep_type] = model.add_parameters((config.sent_dim, 2 * config.sent_dim))

# define trainable projection layer from word dim to phrase dim
# this simplifies concatenation and allows us to treat the recursive base case as a phrase of its own
word_to_phrase_projection = model.add_parameters((config.sent_dim, word_dim))

# define graph building operation
def generate_graph(parse) :
    parse_graph = parse.to_tree()
    return graph_gen_helper(parse_graph)

def graph_gen_helper(node) :
    node_value = word_to_phrase_projection * embeddings[node.data.form]

    for child in node :
        child_subtree = graph_gen_helper(child)

        # concatenate the node so far with the subtree, select layer according to dep reln
        node_value = dep_layers[child.data.deprel] * dynet.concatenate([node_value, child_subtree])

    return node_value

# run training
for parse, y_pred in zip(parse_train, y_preds) :
    y_pred = generate_graph(parse)
    loss = dynet.l1_distance(dynet.l2_norm(y_pred), dynet.l2_norm(y))

# run eval

