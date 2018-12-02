# MATRIX FACTORIZATION

import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_path = os.path.dirname(os.path.realpath(__file__))

L_FACTORS = 32  # latent factors
TRAIN_SIZE = 0.8  # proportion of training set
ITERATIONS = 10000  # number of iterations for optimization (integer)
LEARNING_RATE = 1.0  # learning rate a (float)
L_DECAY_STEP = 1000  # decay step of learning rate
L_DECAY_RATE = 1.0  # decay rate of learning rate
REG_LAMBDA = 0.05  # regularization parameter lambda (float)


def variable_summaries(var, name="summaries"):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        # mean = tf.reduce_mean(var)
        # tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        maxx = tf.summary.scalar('max', tf.reduce_max(var))
        minn = tf.summary.scalar('min', tf.reduce_min(var))
        histt = tf.summary.histogram('histogram', var)
        summaries = tf.summary.merge([maxx, minn, histt])
    return summaries


def get_dataset():
    names = ["user_id", "item_id", "rating", "id"]
    dataset = pd.read_csv(dir_path+'/u.data', sep="\t", names=names)
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


def regulization(Pred, U, I, bu, bi, reg_lambda):
    with tf.name_scope("Regularization"):
        reg_U = tf.multiply(tf.reduce_sum(
            tf.square(U), 1, keepdims=True), reg_lambda)
        reg_I = tf.multiply(tf.reduce_sum(
            tf.square(I), 0, keepdims=True), reg_lambda)
        reg_bu = tf.multiply(tf.square(bu), reg_lambda)
        reg_bi = tf.multiply(tf.square(bi), reg_lambda)
        reg = tf.add(Pred, reg_U)
        reg = tf.add(reg, reg_I)
        reg = tf.add(reg, reg_bu)
        reg = tf. add(reg, reg_bi)
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
        meanU = tf.reduce_mean(U, 1, keepdims=True)
        bu.assign(meanU)
    with tf.name_scope("Mean_I"):
        meanI = tf.reduce_mean(I, 0, keepdims=True)
        bi.assign(meanI)

    R = tf.placeholder(tf.float32, shape=(M, N), name="Train_data")

    with tf.name_scope("Mean"):
        m = tf.reduce_mean(tf.matmul(U, I), name="Mean")

    R_pred = prediction(U, I, bu, bi, m)
    R_pred_reg = regulization(R_pred, U, I, bu, bi, reg_lambda)

    Loss = rmse(R, R_pred_reg, name="Loss")
    RMSE = rmse(R, R_pred, name="RMSE_error")

    rmse_summary = variable_summaries(RMSE, "RMSE")

    # Learning rate decay
    lr = tf.constant(l_rate, name='learning_rate')
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        lr, global_step, l_decay_step, l_decay_rate, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(Loss, global_step=global_step)
    init = tf.global_variables_initializer()
    # merged = tf.summary.merge_all()

    with tf.Session() as sess:
        folder = "%d_%d_%0.3f_%d_%0.2f_%0.4f" % (
            K, iterations, l_rate, l_decay_step, l_decay_rate, reg_lambda)
        train_writer = tf.summary.FileWriter(
            "output/" + folder + "/train", sess.graph)
        test_writer = tf.summary.FileWriter('output/' + folder + '/test')
        sess.run(init, feed_dict={R: data_train})
        feed_train = {R: data_train}
        feed_test = {R: data_test}
        datime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        string_1 = "Latent Factors: %d, Iterations: %d, Learning decay step: %d, Learning decay rate: %0.2f, Regulization parameter: %0.4f" % (
            K, iterations, l_decay_step, l_decay_rate, reg_lambda)
        with open('output/result.csv', 'a') as f:
            f.write("\n\n%s\nStarted: %s" % (string_1, datime))
        print("%s\nStarted: %s" % (string_1, datime))

        rmse_train = sess.run(RMSE, feed_dict=feed_train)
        rmse_test = sess.run(RMSE, feed_dict=feed_test)

        string_2 = "Completed: %0.2f%%,  RMSE train: %0.5f,  RMSE test: %0.5f,  Learning Rate: %f" % (0.0, round(rmse_train, 5),
                                                                                                      round(rmse_test, 5), learning_rate.eval())
        sys.stdout.write("\r%s" % string_2)
        sys.stdout.flush()
        with open('output/result.csv', 'a') as f:
            f.write("\n%s" % string_2)

        for i in range(iterations):
            summary_train, _ = sess.run(
                [rmse_summary, train], feed_dict=feed_train)
            summary_test = sess.run(rmse_summary, feed_dict=feed_test)
            train_writer.add_summary(summary_train, i)
            test_writer.add_summary(summary_test, i)

            if (i + 1) % int(iterations / 2000) == 0:
                rmse_train = sess.run(RMSE, feed_dict=feed_train)
                rmse_test = sess.run(RMSE, feed_dict=feed_test)

                string_2 = "Completed: %0.2f%%,  RMSE train: %0.5f,  RMSE test: %0.5f,  Learning Rate: %f" % ((i + 1) * 100.0 / iterations, round(rmse_train, 5),
                                                                                                              round(rmse_test, 5), learning_rate.eval())
                sys.stdout.write("\r%s" % string_2)
                sys.stdout.flush()
                if (i + 1) % int(iterations / 20) == 0:
                    with open('output/result.csv', 'a') as f:
                        f.write("\n%s" % string_2)
                    # print( rmse(R, prediction(U, I, bu, bi, m)).eval())
        datime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('output/result.csv', 'a') as f:
            f.write("\nFinished: %s" % datime)
        print("\nFinished: %s" % datime)


matrix_factorization(get_dataset(), L_FACTORS, TRAIN_SIZE,
                     ITERATIONS, LEARNING_RATE, REG_LAMBDA, L_DECAY_STEP, L_DECAY_RATE)
