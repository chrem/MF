import tensorflow as tf
import numpy as np
import pandas as pd

x = tf.placeholder(tf.float32, shape=(943, 1682))
y = tf.square(x)
rand_array = np.random.rand(1024, 1024)
names = ["user_id", "item_id", "rating", "id"]
dataset = pd.read_csv('u.data', sep="\t", names=names)
dataset_table = dataset.pivot(
    index=names[0], columns=names[1], values=names[2])
Ratings = np.array(dataset_table, dtype=float)

with tf.Session() as sess:
    # print(sess.run(y))  # ERROR: will fail because x was not fed.
    print(sess.run(y, feed_dict={x: Ratings}))  # Will succeed.
