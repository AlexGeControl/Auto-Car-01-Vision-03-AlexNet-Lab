import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread

from alexnet import AlexNet
import tensorflow.contrib.layers as layers

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

# Resize input images:
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

with tf.variable_scope('alexnet_features'):
    # By setting `feature_extract` to `True` we return the second to last layer.
    # Which is the output of 2nd fully connected layer
    alexnet_features = AlexNet(resized, feature_extract=True)

with tf.variable_scope('fc1'):
    logits = layers.fully_connected(
        alexnet_features, nb_classes,
        activation_fn = tf.identity
    )
    probs = tf.nn.softmax(
        logits
    )

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
