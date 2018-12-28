import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.externals import joblib

DATASET = 'D:/SAT-INSTANCE/SAT12-indu-data'


def normal_data():
    filename = DATASET
    data = joblib.load(filename)
    raw_feature = []
    raw_label = []
    raw_runtime = []
    for i, dic in data:
        raw_feature.append(dic['aim'])
        raw_label.append(dic['index'])
        raw_runtime.append(dic['runtime'])
    print(len(raw_feature), len(raw_label), len(raw_runtime))
    return [raw_feature[:1152], raw_label[:1152], raw_runtime[:1152]]


def get_graph(X, y, gst):
    with tf.variable_scope('One'):
        weight = tf.get_variable(
            name='Weight',
            shape=[3, 3, 1, 32],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        bias = tf.get_variable(
            name='Bias',
            shape=[32],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        conv = tf.nn.conv2d(
            X,
            weight,
            strides=[1, 1, 1, 1],
            padding="SAME",
            name='Convolution_Op')
        act = tf.nn.relu(
            tf.nn.bias_add(conv, bias, data_format='NHWC'), name='Active_Op')
        pool = tf.nn.max_pool(
            act,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='Pooling_Op')
        dropout = tf.nn.dropout(pool, keep_prob=0.9, name='Dropout_Op')

    with tf.variable_scope('Two'):
        weight = tf.get_variable(
            name='Weight',
            shape=[2, 2, 32, 64],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        bias = tf.get_variable(
            name='Bias',
            shape=[64],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        conv = tf.nn.conv2d(
            dropout,
            weight,
            strides=[1, 1, 1, 1],
            padding="SAME",
            name='Convolution_Op')
        act = tf.nn.relu(
            tf.nn.bias_add(conv, bias, data_format='NHWC'), name='Active_Op')
        pool = tf.nn.max_pool(
            act,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='Pooling_Op')
        dropout = tf.nn.dropout(pool, keep_prob=0.8, name='Dropout_Op')

    with tf.variable_scope('Three'):
        weight = tf.get_variable(
            name='Weight',
            shape=[2, 2, 64, 128],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        bias = tf.get_variable(
            name='Bias',
            shape=[128],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        conv = tf.nn.conv2d(
            dropout,
            weight,
            strides=[1, 1, 1, 1],
            padding="SAME",
            name='Convolution_Op')
        act = tf.nn.relu(
            tf.nn.bias_add(conv, bias, data_format='NHWC'), name='Active_Op')
        pool = tf.nn.max_pool(
            act,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='Pooling_Op')
        dropout = tf.nn.dropout(pool, keep_prob=0.7, name='Dropout_Op')

    conv_flat = tf.reshape(dropout, [-1, 16 * 16 * 128])

    with tf.variable_scope("Four"):
        weight = tf.get_variable(
            name='Weight',
            shape=[16 * 16 * 128, 1000],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        bias = tf.get_variable(
            name='Bias',
            shape=[1000],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        act = tf.nn.relu(
            tf.nn.bias_add(tf.matmul(conv_flat, weight), bias), name='Active_Op')
        dropout = tf.nn.dropout(act, keep_prob=0.5, name='Dropout')

    with tf.variable_scope("Five"):
        weight = tf.get_variable(
            name='Weight',
            shape=[1000, 200],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        bias = tf.get_variable(
            name='Bias',
            shape=[200],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        act = tf.nn.relu(
            tf.nn.bias_add(tf.matmul(dropout, weight), bias), name='Active_Op')

    with tf.variable_scope("Six"):
        weight = tf.get_variable(
            name='Weight',
            shape=[200, 31],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        bias = tf.get_variable(
            name='Bias',
            shape=[31],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        out = tf.nn.sigmoid(
            tf.nn.bias_add(tf.matmul(act, weight), bias), name='Active_Op')

    loss = tf.losses.log_loss(labels=y, predictions=out, epsilon=0)
    opt = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9, use_nesterov=True).minimize(loss=loss,
                                                                                                  global_step=gst)
    return opt, loss


fea, label, _ = normal_data()
with ops.Graph().as_default() as g:
    gst = tf.Variable(0, trainable=False, name='global_step')
    # with g.device("/device:GPU:0"):
    X = tf.placeholder(
        dtype=tf.float32, shape=[None, 128, 128, 1], name='Inputs')
    y = tf.placeholder(
        dtype=tf.float32, shape=[None, 31], name='Label')
    opt, loss = get_graph(X, y, gst)
    with g.device('/CPU:0'):
        tf.summary.scalar('loss', loss)
        sum_merge = tf.summary.merge_all()
        gs_op = tf.train.get_global_step()
        saver = tf.train.Saver()
    with tf.Session(graph=g) as ss:
        wrietr = tf.summary.FileWriter("C:/Users/MN/Desktop/td_")
        wrietr.add_graph(ss.graph)

        ss.run(tf.global_variables_initializer())
        for e in range(10):
            for b in range(9):
                b_fea = fea[b * 128:(b + 1) * 128]
                b_lab = label[b * 128:(b + 1) * 128]
                print(b + e * 9)
                _, gs, loss_value, sm = ss.run([opt, gs_op, loss, sum_merge], {X: b_fea, y: b_lab})
                # _, loss_value, sm = ss.run([opt, loss, sum_merge], {X: b_fea, y: b_lab})
                wrietr.add_summary(sm, gs)
            if e == 9 and b == 8:
                saver.save(ss, save_path="C:/Users/MN/Desktop/td_/model.ckpt",
                           global_step=gs)
