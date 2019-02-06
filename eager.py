# A playground for TensorFlow eager execution

import tensorflow as tf
import os

# Used to avoid warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.enable_eager_execution()
print(tf.executing_eagerly())

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))