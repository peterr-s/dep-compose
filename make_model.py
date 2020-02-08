#!/usr/bin/env python3

import sys
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

class Word :
    def __init__(self, line) :
        fields = line.split()
        self.surface = fields[1]
        self.parent = int(fields[6]) if fields[6].isdigit() else 0
        self.reln = fields[7]
        self.children = set()

    def __str__(self) :
        return self.surface

# TODO find better solution
def n(n_str) :
    try :
        return int(n_str)
    except ValueError :
        pass
    return float(n_str)

# read hyperparameters from file
hparams = dict()
with open("./hyperparams.conf", "r") as hyperparameter_file :
    for line in hyperparameter_file.readlines() :
        fields = line.strip().split()
        if len(fields) > 1 :
            hparams[fields[0]] = n(fields[1])
sent_embedding_dim = hparams["sent_embedding_dim"]
hidden_size = hparams["hidden_size"]
learning_rate = hparams["learning_rate"]
epoch_ct = hparams["epoch_ct"]
sigmoid_cutoff = hparams["sigmoid_cutoff"]
dep_embedding_dim = hparams["dep_embedding_dim"]

# load embeddings
embeddings = KeyedVectors.load_word2vec_format(sys.argv[3], binary = False)
word_embedding_dim = embeddings.vector_size
null_word = np.zeros(word_embedding_dim)

# convert embeddings to Keras lookup structure
embedding_mat = np.zeros((len(embeddings.wv.vocab) + 1, word_embedding_dim))
word_to_idx = {embeddings.wv.index2word[i]: i for i in range(len(embeddings.wv.vocab))}
for i in range(len(embeddings.wv.vocab)) :
    e = embeddings.wv[model.wv.index2word[i]]
    embedding_mat[i] = e if e is not None else null_word
embedding_mat[-1] = null_word
word_embedding_layer = tf.keras.Embedding(
        input_dim = embedding_mat.shape[0],
        output_dim = embedding_mat.shape[1],
        weights = [embedding_mat],
        trainable = False
        )

# we need a fixed length for the embedding phrases
# remember, this must be meaningfully less than twice the word embedding dimensionality, else it might just learn to concatenate the vectors
if sent_embedding_dim < (2 * word_embedding_dim) :
    print("warning: too many output dimensions!", file = sys.stderr)

# load training phrases from input file
phrases = list()
dep_to_idx = dict()
idx_to_dep = set()
with open(sys.argv[1], "r") as train_file :
    # phrase structure: [head, tail, dependency, [y_1, y_2, ...]]
    for line in train_file.readlines() :
        fields = line.strip().split()
        phrase = fields[:3] # [head, tail, dependency]
        phrases.append(fields[3:]) # target embedding
        idx_to_dep.add(phrase[2])
idx_to_dep = list(idx_to_dep)
dep_to_idx = {idx_to_dep[i] for i in range(len(idx_to_dep))}

# create dependency embedding lookup structure
dep_embedding_layer = tf.keras.Embedding(
        input_dim = len(idx_to_dep),
        output_dim = dep_embedding_dim
        )

# set up target
y = tf.placeholder(dtype = tf.float32, shape = [sent_embedding_dim, batch_size], name = "y")

# recursive graph building
# this no longer needs to be recursive if we only do bigrams, but any more and it does
# the new idiomatic way to do this is with the Keras Functional API
def compose_embedding(word) :
    tail_layer = tf.keras.layers.concatenate([
            tf.keras.layers.concatenate([
                compose_embedding(child),
                dep_embedding_layer(dep_to_idx[child.reln])
                ])
            for child in word.children
            ]) if len(word.children) > 0 else word_embedding_layer(word_to_idx[word.surface])
    return tf.keras.layers.Dense(sent_embedding_dim)(tail_layer)

if __name__ == "__main__" :
    sess = tf.Session()

    # for each sentence, build a graph and then do a round of training
    preds = list()
    for i, sentence in enumerate(sentences) :
        preds.append(compose_embedding(sentence[0]))
    y_pred = tf.concat(preds, axis = 1)

    with tf.variable_scope("train", reuse = tf.AUTO_REUSE) :
        loss = tf.losses.cosine_distance(tf.math.l2_normalize(y), y_pred, axis = 1)
        loss = tf.identity(loss, name = "loss")
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss, name = "train")

        sess.run(tf.global_variables_initializer())
        for _ in range(epoch_ct) :
                l_val, _ = sess.run([loss, train], {y: np.transpose(sentence_embeddings)})
                print("loss:", l_val)

    # test and print results
    sentences = list()
    with open(sys.argv[2], "r") as test_file :
        sentence = list()

        for line in test_file.readlines() :
            if line[0] == "#" : # ignore commented lines
                continue
            elif line.strip() == "" : # blank lines denote a sentence boundary
                # associate each word except the root with its children
                for i in range(1, len(sentence)) :
                    sentence[sentence[i].parent].children.add(sentence[i])
                # add to training set
                sentences.append(sentence)

                sentence = [Word("0 %ROOT %ROOT - - - 0 %ROOT - -")]
            else :
                sentence.append(Word(line))

        if not sentences[-1] == sentence : # guard against lack of trailing newline
            sentences.append(sentence)
    sentence_embeddings = skipthoughts_encoder.encode([" ".join([w.surface for w in s]) for s in sentences])

    pred = list()
    avg_loss = 0
    for i in range(len(sentences) // batch_size) :
        batch_yhat, batch_loss = sess.run([y_pred, loss], {y: np.transpose(sentence_embeddings[(i * batch_size) : ((i + 1) * batch_size)])})
        pred.append(np.transpose(batch_yhat))
        avg_loss += batch_loss
    avg_loss /= len(sentences) // batch_size
    print(avg_loss)

    with open("%s_sentences.txt" % sys.argv[2], "w+") as sentence_output_file :
        for sentence in sentences :
            print(" ".join([w.surface for w in sentence]), file = sentence_output_file)

    with open("%s.skipthoughts" % sys.argv[2], "w+") as skipthoughts_output_file :
        for embedding in sentence_embeddings :
            for x in embedding :
                print(x, end = " ", file = skipthoughts_output_file)
            print(file = skipthoughts_output_file)

    with open("%s.recursion" % sys.argv[2], "w+") as recursion_output_file :
        for batch in pred :
            batch = np.reshape(batch, (batch_size, sent_embedding_dim))
            for embedding in batch :
                for x in embedding :
                    print(x, end = " ", file = recursion_output_file)
                print(file = recursion_output_file)

