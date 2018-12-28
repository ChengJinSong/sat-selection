""" cnn model based on SAT problem """
# pylint: disable=C0103,C0111
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# def cnn_model(features, labels, mode):
#     """ building a concrete cnn model function """

#     # get a input layer by features and labels, its size is 128*128
#     input_layer = []

#     # first convolutional layer
#     conv1 = tf.layers.conv2d(
#         inputs=input_layer,
#         filters=32,
#         kernel_size=[3, 3],
#         strides=(1, 1),
#         padding='same',
#         activation=tf.nn.relu)
#     pooling1 = tf.layers.max_pooling2d(
#         inputs=conv1, pool_size=[2, 2], strides=1, padding='same')
#     dropout1 = tf.layers.dropout(inputs=pooling1, rate=0.1)

#     # second convolutional layer
#     conv2 = tf.layers.conv2d(
#         inputs=dropout1,
#         filters=64,
#         kernel_size=[2, 2],
#         strides=(1, 1),
#         padding='same',
#         activation=tf.nn.relu)
#     pooling2 = tf.layers.max_pooling2d(
#         inputs=conv2, pool_size=[2, 2], strides=1, padding='same')
#     dropout2 = tf.layers.dropout(inputs=pooling2, rate=0.2)

#     # third convolutinal layer
#     conv3 = tf.layers.conv2d(
#         inputs=dropout2,
#         filters=128,
#         kernel_size=[2, 2],
#         strides=(1, 1),
#         padding='same',
#         activation=tf.nn.relu)
#     pooling3 = tf.layers.max_pooling2d(
#         inputs=conv3, pool_size=[2, 2], strides=1, padding='same')
#     dropout3 = tf.layers.dropout(inputs=pooling3, rate=0.3)

#     # fully connected layer
#     dense1 = tf.layers.dense(inputs=dropout3, units=1000)


# ================================================================

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "labels": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy":
        tf.metrics.accuracy(labels=labels, predictions=predictions["labels"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
