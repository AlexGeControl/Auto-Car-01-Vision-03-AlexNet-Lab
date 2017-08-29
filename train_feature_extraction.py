from os.path import join, dirname
from datetime import datetime
import time

import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Generate more data to feed the network
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from alexnet import AlexNet
import tensorflow.contrib.layers as layers

# Load traffic signs data.
with open('train.p', 'rb') as f:
    train = pickle.load(f)

# Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(
    train["features"], train["labels"],
    test_size = 0.20, random_state = 42
)

N_CLASSES = 43

image_data_generator = ImageDataGenerator(
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    zoom_range = 0.25,
    fill_mode='nearest'
)

# Define placeholders and resize operation:
with tf.variable_scope('input'):
    X = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
    resized = tf.image.resize_images(X, (227, 227))
    y = tf.placeholder(shape=(None), dtype=tf.int64)

# Extract features using 'AlexNet':
with tf.variable_scope('alexnet_features'):
    alexnet_features = AlexNet(resized, feature_extract=True)
    # `tf.stop_gradient` prevents the gradient from flowing backwards past this point,
    # keeping the weights before and up to `alexnet_features` frozen.
    alexnet_features = tf.stop_gradient(alexnet_features)

# Add the final layer for traffic sign classification.
with tf.variable_scope('output'):
    logits = layers.fully_connected(
        alexnet_features, N_CLASSES,
        activation_fn = tf.identity
    )
    probs = tf.nn.softmax(
        logits
    )

# Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y,
        logits=logits
    )
    loss = tf.reduce_mean(cross_entropy)

with tf.name_scope('optimization'):
    # Learning rate scheduling:
    global_step = tf.Variable(
        0, dtype=tf.int32, trainable=False
    )

    BOUNDARIES = [
        1000
    ]
    LEARNING_RATES = [
        1e-3, 1e-4
    ]

    # Piecewise constant:
    learning_rate = tf.train.piecewise_constant(
        global_step,
        BOUNDARIES,
        LEARNING_RATES
    )

    # Adam optimizer:
    optimizer = tf.train.AdamOptimizer(
        learning_rate
    ).minimize(
        loss,
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output/fully_connected'),
        global_step = global_step
    )

with tf.name_scope('evaluation'):
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(logits, 1), y),
            tf.float32
        )
    )

logdir = join(
    "tf-logs",
    "run-{}".format(
        datetime.utcnow().strftime("%Y%m%d%H%M%S")
    )
)

with tf.name_scope('logs'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar("accuracy", accuracy)
    summary = tf.summary.merge_all()

logger = tf.summary.FileWriter(
    logdir,
    tf.get_default_graph()
)

saver = tf.train.Saver()

BATCH_SIZE = 512
SKIP_STEP = 100

MAX_EPOCHES = 10
MAX_ITERS = 2000

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Whether pre-trained network exists:
    latest_checkpoint = tf.train.get_checkpoint_state(
        'checkpoints'
    )
    # If that checkpoint exists, restore from checkpoint
    if latest_checkpoint and latest_checkpoint.model_checkpoint_path:
        saver.restore(sess, latest_checkpoint.model_checkpoint_path)
        print("Load pre-trained network")
    # Else initialize all variables:
    else:
        init.run()
        print("Start from scratch")

    # Initialize stats:
    start_time = time.time()
    total_loss = 0.0
    INIT_ITERS = global_step.eval()

    # Current mini-batch:
    for X_batch, y_batch in image_data_generator.flow(X_train, y_train, batch_size=BATCH_SIZE):
        # Iteration index:
        iter_index = global_step.eval() + 1

        # Stop training when reaching max. num. of iterations:
        if iter_index > MAX_ITERS + INIT_ITERS:
            break

        # Train:
        _, loss_batch = sess.run(
            [optimizer, loss],
            feed_dict = {
                X: X_batch,
                y: y_batch
            }
        )

        total_loss += loss_batch

        if iter_index % SKIP_STEP == 0:
            # Training set:
            logger.add_summary(
                summary.eval(
                    feed_dict = {
                        X: X_batch,
                        y: y_batch
                    }
                ),
                global_step = iter_index
            )

            # Testing set:
            M_SAMPLES = X_test.shape[0]
            NUM_BATCHES = int(np.ceil(M_SAMPLES / BATCH_SIZE))

            test_accuracy = 0.0

            for batch in range(NUM_BATCHES):
                X_test_batch = X_test[batch*BATCH_SIZE: (batch + 1)*BATCH_SIZE]
                y_test_batch = y_test[batch*BATCH_SIZE: (batch + 1)*BATCH_SIZE]
                test_accuracy_batch = accuracy.eval(
                    feed_dict = {
                        X: X_test_batch,
                        y: y_test_batch
                    }
                )
                test_accuracy += len(X_test_batch) * test_accuracy_batch
            test_accuracy /= M_SAMPLES

            print(
                "[Performance @ step {}]: {:2.4f}-{:2.4f}".format(
                    iter_index,
                    total_loss / SKIP_STEP,
                    test_accuracy
                )
            )
            total_loss = 0.0
            saver.save(sess, "checkpoints/traffic-sign-{}.ckpt".format(iter_index))

    saver.save(sess, "checkpoints/traffic-sign-{}.ckpt".format("final"))

    print("Optimization Finished!")
    print("Total time: {0} seconds".format(time.time() - start_time))

logger.close()
