import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

K = 5  # latent factors
TRAIN_SIZE = 0.8  # proportion of training set
ITERATIONS = 100000  # number of iterations for optimazation
LEARNING_RATE = 0.0003  # learning rate a
REG_LAMBDA = 0.3  # regulization parameter lambda


def split_dataset(x):
    mask = ~np.isnan(x)
    i, j = np.nonzero(mask)
    selection = np.zeros(len(i), dtype=bool)
    p = int(len(i) * TRAIN_SIZE)
    selection[:p] = 1
    np.random.shuffle(selection)
    mask[i, j] = selection
    train = np.array(x)
    test = np.array(x)
    train[~mask] = np.nan
    test[mask] = np.nan
    return train, test


names = ["user_id", "item_id", "rating", "id"]
dataset = pd.read_csv('u.data', sep="\t", names=names)
dataset_table = dataset.pivot(
    index=names[0], columns=names[1], values=names[2])
Ratings = np.array(dataset_table)
R_train, R_test = split_dataset(Ratings)


def isnt_nan(x):
    y = tf.logical_not(tf.is_nan(x))
    return y


def rmse(x, y):
    # square_error = tf.squared_difference(x, y)
    square_error = tf.square(x - y)
    boolean_mask = tf.boolean_mask(square_error, isnt_nan(square_error))
    mean_error = tf.reduce_mean(boolean_mask)
    root_mse = tf.sqrt(mean_error)
    return root_mse


def ML(data):
    M = len(data)
    N = len(data[0])
    U = tf.get_variable("Users", shape=[M, K])
    I = tf.get_variable("Items", shape=[K, N])
    # nans = tf.ones([M, N])
    nans = tf.constant(float('NaN'), shape=[M, N])
    # bu = tf.get_variable("User_bias", shape=[M, 1])
    # bi = tf.get_variable("Item_bias", shape=[1, N])
    # R = tf.placeholder(tf.float32, shape=[M, N], name="Ratings")
    R = tf.Variable(data, dtype=tf.float32)

    R_pred_1 = tf.matmul(U, I)
    R_pred = tf.where(isnt_nan(R), R_pred_1, nans)
    RMSE = rmse(R, R_pred)
    #RMSE = tf.square(R - R_pred)

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train = optimizer.minimize(RMSE, var_list=[U, I])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in xrange(ITERATIONS):
            sys.stdout.write("\r%d %0.5f" % (i, round(RMSE.eval(), 5)))
            sys.stdout.flush()
            sess.run(train)
        print"\n"
        print R_pred.eval()
        print R.eval()


a = np.array([[1, 2, 3, 4, 5], [2, np.nan, 5, 6, 7], [8, 7, 6, 5, np.nan]])
ML(a)
