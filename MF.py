import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

L_FACTORS = 32  # latent factors
TRAIN_SIZE = 0.8  # proportion of training set
ITERATIONS = 1000  # number of iterations for optimazation
LEARNING_RATE = 50.0  # learning rate a
REG_LAMBDA = 0.5  # regulization parameter lambda


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


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


def rmse(x, y, name="RMSE"):
    with tf.name_scope(name):
        nonan_x = tf.boolean_mask(x, isnt_nan(x))
        nonan_y = tf.boolean_mask(y, isnt_nan(x))
        square_error = tf.squared_difference(nonan_x, nonan_y)
        mean_error = tf.reduce_mean(square_error)
        root_mse = tf.sqrt(mean_error)
    return root_mse


def prediction(U, I, bu, bi, m):
    with tf.name_scope("Prediction"):
        pred = tf.matmul(U, I)
        pred = tf.add(pred, bu)
        pred = tf.add(pred, bi)
        pred = tf.add(pred, m)
    return pred


def ML(data, K, train_size=0.8, iterations=5000, l_rate=0.03):
    M = len(data)
    N = len(data[0])
    data_train, data_test = split_dataset(data, train_size)

    U = tf.get_variable("Users", shape=(M, K))
    I = tf.get_variable("Items", shape=(K, N))
    bu = tf.get_variable("User_bias", shape=(M, 1))
    bi = tf.get_variable("Item_bias", shape=(1, N))
    with tf.name_scope("Mean_U"):
        meanU = tf.stack([tf.reduce_mean(U, 1)])
        bu.assign(tf.transpose(meanU))
    with tf.name_scope("Mean_I"):
        meanI = tf.stack([tf.reduce_mean(I, 0)])
        bi.assign(meanI)

    # variable_summaries(U)
    # variable_summaries(I)
    # variable_summaries(bu)
    # variable_summaries(bi)

    R_train = tf.placeholder(tf.float32, shape=(M, N), name="Train_data")
    #R_test = tf.placeholder(tf.float32, shape=(M, N), name="Test_data")

    # R_train = tf.Variable(data_train, dtype=tf.float32, name="Train_data", trainable=False)
    # R_test = tf.Variable(data_test, dtype=tf.float32, name = "Test_data", trainable = False)

    with tf.name_scope("Mean"):
        m = tf.reduce_mean(tf.matmul(U, I), name="Mean")

    R_pred = prediction(U, I, bu, bi, m)

    Loss = rmse(R_train, R_pred, name="Loss")
    # RMSE_test = rmse(R_test, R_pred, name="RMSE_test")
    tf.summary.histogram('Loss', Loss)
    # tf.summary.histogram('RMSE_test', RMSE_test)

    lr = tf.constant(l_rate, name='learning_rate')
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        lr, global_step, 10, 0.96, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(Loss)
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("train", sess.graph)
        # test_writer = tf.summary.FileWriter('test')
        sess.run(init, feed_dict={R_train: data_train})
        for i in xrange(iterations):
            summary, _ = sess.run([merged, train], feed_dict={
                R_train: data_train})
            # sess.run(train)
            train_writer.add_summary(summary, i)

            sys.stdout.write("\rCompleted: %0.2f%%" %
                             ((i + 1) * 100.0 / iterations))
            sys.stdout.flush()
            # print rmse(R_train, prediction(U, I, bu, bi, m)).eval()
        print"\n"


names = ["user_id", "item_id", "rating", "id"]
dataset = pd.read_csv('u.data', sep="\t", names=names)

dataset_table = dataset.pivot(
    index=names[0], columns=names[1], values=names[2])
Ratings = np.array(dataset_table, dtype=float)


ML(Ratings, L_FACTORS, TRAIN_SIZE, ITERATIONS, LEARNING_RATE)
