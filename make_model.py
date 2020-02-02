#!/usr/bin/env python3

import sys
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import skipthoughts

class Word :
    def __init__(self, line) :
        fields = line.split()
        self.surface = fields[1]
        self.parent = int(fields[6]) if fields[6].isdigit() else 0
        self.reln = fields[7]
        self.children = set()

    def __str__(self) :
        return self.surface

sent_embedding_dim = 4800
hidden_size = 9600
max_sent_len = 5
learning_rate = 0.2
epoch_ct = 200
sigmoid_cutoff = 8

# this prevents the weights from being perpetually adjusted after they're good enough
def cutoff_sigmoid(x, name = None) :
    pos_mask = tf.where(
            tf.greater(x, sigmoid_cutoff),
            x = tf.fill(x.shape, 1.0),
            y = tf.sigmoid(x)
            )
    neg_mask = tf.where(
            tf.less(pos_mask, -sigmoid_cutoff),
            x = tf.fill(x.shape, 0.0),
            y = pos_mask,
            name = name
            )
    return neg_mask

# load training sentences from input file
sentences = list()
with open(sys.argv[1], "r") as train_file :
    sentence = list()

    for line in train_file.readlines() :
        if line[0] == "#" : # ignore commented lines
            continue
        elif line.strip() == "" : # blank lines denote a sentence boundary
            if len(sentence) <= max_sent_len :
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

# load embeddings
embeddings = KeyedVectors.load_word2vec_format(sys.argv[3], binary = False)
word_embedding_dim = embeddings.vector_size
null_word = np.zeros(word_embedding_dim)
#null_word = np.random.rand(word_embedding_dim)

skipthoughts_model = skipthoughts.load_model()
skipthoughts_encoder = skipthoughts.Encoder(skipthoughts_model)

sentence_embeddings = skipthoughts_encoder.encode([" ".join([w.surface for w in s]) for s in sentences])
batch_size = sentence_embeddings.shape[0]

# set up target
y = tf.placeholder(dtype = tf.float32, shape = [sent_embedding_dim, batch_size], name = "y")

# transform word embeddings to sentence dimensionality
t = tf.get_variable("t", dtype = tf.float32, shape = [sent_embedding_dim, word_embedding_dim], trainable = True)

# recursive graph building
def compose_embedding(word) :
    with tf.variable_scope("compose", reuse = tf.AUTO_REUSE) :
        x = tf.Variable(
                tf.transpose(
                    [list(embeddings[word.surface] if word.surface in embeddings else null_word)]
                    ),
                trainable = False
                ) if word.surface is not "%ROOT" else tf.get_variable("x_ROOT", trainable = False) # make the embedding untrainable EXCEPT for root
        x = tf.matmul(t, x)
        for child in word.children :
            w1 = tf.get_variable(
                    ("w1_%s" % child.reln),
                    dtype = tf.float32,
                    shape = [hidden_size, x.shape[0]],
                    trainable = True
                    )
            b1 = tf.get_variable(
                    ("b1_%s" % child.reln),
                    dtype = tf.float32,
                    shape = [hidden_size, 1],
                    trainable = True
                    )
            w2 = tf.get_variable(
                    ("w2_%s" % child.reln),
                    dtype = tf.float32,
                    shape = [sent_embedding_dim, hidden_size],
                    trainable = True
                    )
            b2 = tf.get_variable(
                    ("b2_%s" % child.reln),
                    dtype = tf.float32,
                    shape = [sent_embedding_dim, 1],
                    trainable = True
                    )
            
            x = tf.math.l2_normalize(
                        cutoff_sigmoid(
                        tf.matmul(
                            w2,
                            cutoff_sigmoid(
                                tf.matmul(
                                    w1,
                                    x
                                    ) + b1
                                )
                            ) + b2
                        )
                    )

        return x

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

