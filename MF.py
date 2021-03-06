# MATRIX FACTORIZATION

import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

L_FACTORS = 32  # latent factors
TRAIN_SIZE = 0.8  # proportion of training set
ITERATIONS = 1000  # number of iterations for optimazation (integer)
LEARNING_RATE = 3.0  # starting learning rate a (float)
L_DECAY_STEP = 50.0  # decay step of leraning rate (float)
L_DECAY_RATE = 0.8  # decay rate of learning rate (float)
REG_LAMBDA = 0  # regulization parameter lambda (float)


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


def get_dataset():
    names = ["user_id", "item_id", "rating", "id"]
    dataset = pd.read_csv('u.data', sep="\t", names=names)
    dataset_table = dataset.pivot(
        index=names[0], columns=names[1], values=names[2])
    Ratings = np.array(dataset_table, dtype=float)
    return Ratings


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


def regulization(U, I, bu, bi, reg_lambda):
    reg_U = tf.nn.l2_loss(U)
    # reg_U = tf.multiply(reg_lambda, U2)
    reg_I = tf.nn.l2_loss(I)
    # reg_I = tf.multiply(reg_lambda, I2)
    reg_bu = tf.nn.l2_loss(bu)
    # reg_bu = tf.multiply(reg_lambda, bu2)
    reg_bi = tf.nn.l2_loss(bi)
    # reg_bi = tf.multiply(reg_lambda, bi2)
    reg = tf.add(reg_U, reg_I)
    reg = tf.add(reg, reg_bu)
    reg = tf. add(reg, reg_bi)
    reg = tf.multiply(reg, reg_lambda)
    return reg


def matrix_factorization(data, K, train_size=0.8, iterations=5000, l_rate=0.03, reg_lambda=0.5, l_decay_step=1000, l_decay_rate=0.96):
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

    variable_summaries(U)
    variable_summaries(I)
    variable_summaries(bu)
    variable_summaries(bi)

    R_train = tf.placeholder(tf.float32, shape=(M, N), name="Train_data")
    R_test = tf.placeholder(tf.float32, shape=(M, N), name="Test_data")
    # R_train = tf.Variable(data_train, dtype=tf.float32, name="Train_data", trainable=False)
    # R_test = tf.Variable(data_test, dtype=tf.float32, name = "Test_data", trainable = False)

    with tf.name_scope("Mean"):
        m = tf.reduce_mean(tf.matmul(U, I), name="Mean")

    R_pred = prediction(U, I, bu, bi, m)
    R_pred_reg = R_pred + regulization(U, I, bu, bi, reg_lambda)

    Loss = rmse(R_train, R_pred_reg, name="Loss")
    RMSE_test = rmse(R_test, R_pred, name="RMSE_test")
    tf.summary.histogram('Loss', Loss)
    tf.summary.histogram('RMSE_test', RMSE_test)

    # Learning rate decay
    lr = tf.constant(l_rate, name='learning_rate')
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        lr, global_step, l_decay_step, l_decay_rate, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(Loss, global_step=global_step)
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("output/train", sess.graph)
        # test_writer = tf.summary.FileWriter('test')
        sess.run(init, feed_dict={R_train: data_train})
        feed = {R_train: data_train, R_test: data_test}

        datime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        string_1 = "Latent Factors: %d, Iterations: %d, Learning Rate: %0.3f, Learning decay step: %d, Learning decay rate: %0.2f, Regulization parameter: %0.4f" % (
            K, iterations, learning_rate.eval(), l_decay_step, l_decay_rate, reg_lambda)
        with open('output/result.csv', 'a') as f:
            f.write("\n\n%s\nStarted: %s" % (string_1, datime))
        print "Started: %s" % datime

        for i in xrange(iterations):
            summary, _ = sess.run([merged, train], feed_dict=feed)
            train_writer.add_summary(summary, i)
            if (i + 1) % int(iterations / 1000) == 0:
                rmse_train, rmse_test = sess.run(
                    [Loss, RMSE_test], feed_dict=feed)
                string_2 = "Completed: %0.2f%%,  RMSE train: %0.5f,  RMSE test: %0.5f,  Learning rate: %f" % ((i + 1) * 100.0 / iterations, round(rmse_train, 5),
                                                                                                              round(rmse_test, 5), learning_rate.eval())
                sys.stdout.write("\r%s" % string_2)
                sys.stdout.flush()
                if (i + 1) % int(iterations / 50) == 0:
                    with open('output/result.csv', 'a') as f:
                        f.write("\n%s" % string_2)

        datime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('output/result.csv', 'a') as f:
            f.write("\nFinished: %s" % datime)
        print"\nFinished: %s" % datime


matrix_factorization(get_dataset(), L_FACTORS, TRAIN_SIZE,
                     ITERATIONS, LEARNING_RATE, REG_LAMBDA, L_DECAY_STEP, L_DECAY_RATE)
