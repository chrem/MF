import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
K = 2
iterations = 100


def isnt_nan(x):
    y = tf.logical_not(tf.is_nan(x))
    return y


def rmse(x, y, name="RMSE"):
    with tf.name_scope(name):
        nonan_x = tf.boolean_mask(x, isnt_nan(x))
        nonan_y = tf.boolean_mask(y, isnt_nan(x))
        square_error = tf.squared_difference(nonan_x, nonan_y)
        mean_error = tf.reduce_mean(square_error)
        root_mse = tf.sqrt(mean_error)
    return root_mse


def prediction(U, I):
    with tf.name_scope("Prediction"):
        pred = tf.matmul(U, I)
        # pred = tf.add(pred, bu)
        # pred = tf.add(pred, bi)
        # pred = tf.add(pred, m)
    return pred


names = ["user_id", "item_id", "rating", "id"]
dataset = pd.read_csv('u.data', sep="\t", names=names)
dataset_table = dataset.pivot(
    index=names[0], columns=names[1], values=names[2])
R_data = np.array(dataset_table, dtype=float)

M = len(R_data)
N = len(R_data[0])
R = tf.placeholder(tf.float32, shape=(M, N))
U = tf.get_variable("User", shape=(M, K))
I = tf.get_variable("Item", shape=(K, N))
Y = prediction(U, I)

#rand_array = np.random.rand(np.shape(R))

loss = rmse(R, Y)

optimizer = tf.train.GradientDescentOptimizer(0.03)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in xrange(iterations):
        sess.run(train, feed_dict={R: R_data})
        sys.stdout.write("\rCompleted: %0.2f%%" %
                         ((i + 1) * 100.0 / iterations))
        sys.stdout.flush()

    print(Y.eval())  # Will succeed.
