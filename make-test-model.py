#!/usr/bin/env python3
import tensorflow as tf

import tensorflow as tf
import numpy as np

input_data = np.full((1,480,640,3), 0.5)

x = tf.placeholder(dtype=tf.float32, shape=(1,480,640,3), name='input')
c = tf.constant(0.3, dtype=tf.float32, name='c42')
y = tf.compat.v1.multiply(x, c, name='output')

with tf.Session() as sess:
  result = sess.run(y, feed_dict={x: input_data})
  print(result.shape)

  output_graph_def = tf.get_default_graph().as_graph_def()
  with tf.gfile.GFile("test.pb", "wb") as f:
    f.write(output_graph_def.SerializeToString())
