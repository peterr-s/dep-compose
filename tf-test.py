#!/usr/bin/env python3.7

import tensorflow as tf
import numpy as np

batch_size = 10

embed_dim = 3
x1 = np.random.rand(50000, 1, embed_dim)
x2 = np.random.rand(50000, 1, embed_dim)
y = x1 + x2

a = tf.keras.Input(shape = (1, embed_dim))
b = tf.keras.Input(shape = (1, embed_dim))

conc = tf.keras.layers.concatenate([a, b], axis = 1)
perm = tf.keras.layers.Permute((2, 1))(conc)
conv = tf.keras.layers.Conv1D(filters = 50, kernel_size = embed_dim)(perm)
output = tf.keras.layers.Dense(embed_dim)(conv)

model = tf.keras.Model(inputs = [a, b], outputs = (output))

model.compile(optimizer = tf.keras.optimizers.SGD(), loss = tf.keras.losses.MeanSquaredError())

model.fit([x1, x2], y, batch_size = batch_size, epochs = 3)

y_pred = model.predict((np.array([[[0.1, 0.2, 0.3]], [[0.5, 0.5, 0.6]]]), np.array([[[0.2, 0.3, 0.4]], [[0.01, 0.2, 0]]])))
print(y_pred)

