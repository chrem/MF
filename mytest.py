import numpy
import tensorflow as tf
import pandas as pd
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# read data
df = pd.read_csv('u2.data', sep=',', names=['user', 'item', 'rate'])
msk = numpy.random.rand(len(df)) < 0.7
df_train = df[msk]

user_indecies = [x - 1 for x in df_train.user.values]
item_indecies = [x - 1 for x in df_train.item.values]
rates = df_train.rate.values
# variables
feature_len = 10
U = tf.Variable(initial_value=tf.truncated_normal(
    [943, feature_len]), name='users')
P = tf.Variable(initial_value=tf.truncated_normal(
    [feature_len, 1682]), name='items')
result = tf.matmul(U, P)
result_flatten = tf.reshape(result, [-1])

# rating
R = tf.gather(result_flatten, user_indecies * tf.shape(result)
              [1] + item_indecies, name='extracting_user_rate')

# cost function
diff_op = tf.subtract(R, rates, name='trainig_diff')
diff_op_squared = tf.abs(diff_op, name="squared_difference")
base_cost = tf.reduce_sum(diff_op_squared, name="sum_squared_error")
tf.summary.histogram('Cost', base_cost)

# # regularization
# lda = tf.constant(.001, name='lambda')
# norm_sums = tf.add(tf.reduce_sum(tf.abs(U, name='user_abs'), name='user_norm'),
#                    tf.reduce_sum(tf.abs(P, name='item_abs'), name='item_norm'))
# regularizer = tf.multiply(norm_sums, lda, 'regularizer')

# cost function
lr = tf.constant(.001, name='learning_rate')
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    lr, global_step, 10000, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_step = optimizer.minimize(base_cost, global_step=global_step)

# execute
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("mytest/train", sess.graph)
for i in xrange(1000):
    # sess.run(training_step)
    summary, _ = sess.run([merged, training_step])
    sys.stdout.write("\rCompleted: %0.2f%%,  Cost: %0.5f" %
                     ((i + 1) * 100.0 / 1000, round(sess.run(base_cost), 5)))
    sys.stdout.flush()
print"\n"

# example user=278 item=603 r=5 (from u.data)
u, p, r = df[['user', 'item', 'rate']].values[0]
rhat = tf.gather(tf.gather(result, u - 1), p - 1)
print "rating for user " + str(u) + " for item " + str(p) + " is " + str(r) + " and our prediction is: " + str(sess.run(rhat))


# calculate accuracy
df_test = df[~msk]
user_indecies_test = [x - 1 for x in df_test.user.values]
item_indecies_test = [x - 1 for x in df_test.item.values]
rates_test = df_test.rate.values

# accuracy
R_test = tf.gather(result_flatten, user_indecies_test * tf.shape(result)
                   [1] + item_indecies_test, name='extracting_user_rate_test')
diff_op_test = tf.subtract(R_test, rates_test, name='test_diff')
diff_op_squared_test = tf.abs(diff_op, name="squared_difference_test")

cost_test = tf.div(tf.reduce_sum(tf.square(diff_op_squared_test, name="squared_difference_test"),
                                 name="sum_squared_error_test"), df_test.shape[0] * 2, name="average_error")
print sess.run(cost_test)
