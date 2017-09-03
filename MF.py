import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

L_FACTORS = 16  # latent factors
TRAIN_SIZE = 0.8  # proportion of training set
ITERATIONS = 5000  # number of iterations for optimazation
LEARNING_RATE = 5  # learning rate a
REG_LAMBDA = 0.3  # regulization parameter lambda


def split_dataset(x, train_size):
    mask = ~np.isnan(x)
    i, j = np.nonzero(mask)
    selection = np.zeros(len(i), dtype=bool)
    p = int(len(i) * train_size)
    selection[:p] = 1
    np.random.shuffle(selection)
    mask[i, j] = selection
    train = np.array(x)
    test = np.array(x)
    train[~mask] = np.nan
    test[mask] = np.nan
    return train, test


def isnt_nan(x):
    y = tf.logical_not(tf.is_nan(x))
    return y


def rmse(x, y):
    with tf.name_scope("RMSE"):
        # square_error = tf.squared_difference(x, y)
        square_error = tf.square(x - y)
        boolean_mask = tf.boolean_mask(square_error, isnt_nan(square_error))
        mean_error = tf.reduce_mean(boolean_mask)
        root_mse = tf.sqrt(mean_error)
    return root_mse


def prediction(U, I):
    pred = tf.matmul(U, I)
    return pred


def ML(data, K, train_size=0.8, iterations=5000, learning_rate=0.03):
    M = len(data)
    N = len(data[0])
    data_train, data_test = split_dataset(data, train_size)
    U = tf.get_variable("Users", shape=[M, K])
    I = tf.get_variable("Items", shape=[K, N])
    nans = tf.constant(float('NaN'), shape=[M, N])
    # bu = tf.get_variable("User_bias", shape=[M, 1])
    # bi = tf.get_variable("Item_bias", shape=[1, N])
    # R_train = tf.placeholder(tf.float32, shape=[M, N], name="Ratings")
    R_train = tf.Variable(data_train, dtype=tf.float32)
    R_test = tf.Variable(data_test, dtype=tf.float32)

    R_pred_1 = prediction(U, I)
    R_pred = tf.where(isnt_nan(R_train), R_pred_1, nans)
    RMSE_train = rmse(R_train, R_pred)
    RMSE_test = rmse(R_test, R_pred_1)
    tf.summary.histogram('RMSE_train', RMSE_train)
    tf.summary.histogram('RMSE_test', RMSE_test)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(RMSE_train, var_list=[U, I])

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("train", sess.graph)
        test_writer = tf.summary.FileWriter('/test')
        sess.run(tf.global_variables_initializer())
        for i in xrange(iterations):
            summary, _ = sess.run([merged, train])
            train_writer.add_summary(summary, i)
            # print R.eval()
            sys.stdout.write("\rIteration: %d%%,  RMSE train: %0.5f,  RMSE test: %0.5f" %
                             ((i + 1) * 100.0 / iterations, round(RMSE_train.eval(), 5), round(RMSE_test.eval(), 5)))
            sys.stdout.flush()
        print"\n"
        # print R_pred.eval()


a = np.array([[1, 2, 3, 4, 5], [2, np.nan, 5, 6, 7],
              [8, 7, 6, 5, np.nan]], dtype=float)
names = ["user_id", "item_id", "rating", "id"]
dataset = pd.read_csv('u.data', sep="\t", names=names)
dataset_table = dataset.pivot(
    index=names[0], columns=names[1], values=names[2])
Ratings = np.array(dataset_table)
ML(Ratings, L_FACTORS, TRAIN_SIZE, ITERATIONS, LEARNING_RATE)
