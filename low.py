# pylint: disable=C0103,C0111,W0102,R0913
# pylint: disable=too-many-locals

import get_root
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from tensorflow.python.framework import ops
from sklearn.utils import shuffle

tf.logging.set_verbosity(tf.logging.INFO)

PATH_ROOT = get_root.PATH_ROOT
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 130
LEARNING_RATE = 0.1
MOMENTUM = 0.95
NUM_LABEL = 31
NUM_EPOCH = 5
NUM_BATCH_OF_EPOCHS = 8
TRAIN_EVAL = False
TF_MODEL_PATH = PATH_ROOT + 'tmp_model/_test/'
IMG_SIZE = [128, 128]
DATASET = PATH_ROOT + 'SAT-INSTANCE/SAT12-indu-data'


def conv_layer(inputs,
               kernel_shape,
               kernel_stride=[1, 1, 1, 1],
               activition=tf.nn.relu,
               pool_ksize=[1, 2, 2, 1],
               pool_stride=[1, 2, 2, 1],
               keep_rate=0.9,
               name="Conv"):
    """
    input shape: [batch, height, width, channel]
    kernel shape: [k_height, k_width, k_channel, k_count]
    bias shape: [k_count]
    """
    with tf.variable_scope(name):
        weight = tf.get_variable(
            name='Weight',
            shape=kernel_shape,
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        bias = tf.get_variable(
            name='Bias',
            shape=[kernel_shape[-1]],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        conv = tf.nn.conv2d(
            inputs,
            weight,
            strides=kernel_stride,
            padding="SAME",
            name='Convolution_Op')
        act = activition(
            tf.nn.bias_add(conv, bias, data_format='NHWC'), name='Active_Op')
        pool = tf.nn.max_pool(
            act,
            ksize=pool_ksize,
            strides=pool_stride,
            padding='SAME',
            name='Pooling_Op')
        with tf.device('/CPU:0'):
            s_wei_name = '{}_weight'.format(name)
            s_bi_name = '{}_bias'.format(name)
            tf.summary.histogram(s_wei_name, weight)
            tf.summary.histogram(s_bi_name, bias)
        return tf.nn.dropout(pool, keep_prob=keep_rate, name='Dropout_Op')


def fc_layer(inputs,
             in_neurons,
             out_neurons,
             activition=tf.nn.relu,
             name='FullConnect'):
    """
    inputs: shape of [batch, in_neurons, 1]
    """
    with tf.variable_scope(name):
        weight = tf.get_variable(
            name='Weight',
            shape=[in_neurons, out_neurons],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        bias = tf.get_variable(
            name='Bias',
            shape=[out_neurons],
            initializer=tf.glorot_normal_initializer(dtype=tf.float32))
        with tf.device('/CPU:0'):
            s_wei_name = '{}_weight'.format(name)
            s_bi_name = '{}_bias'.format(name)
            tf.summary.histogram(s_wei_name, weight)
            tf.summary.histogram(s_bi_name, bias)
        return activition(
            tf.nn.bias_add(tf.matmul(inputs, weight), bias), name='Active_Op')


def normal_data():
    filename = DATASET
    data = joblib.load(filename)
    raw_feature = []
    raw_label = []
    raw_runtime = []
    for i, dic in data:
        raw_feature.append(dic['aim'])  # instance array image shape of [128,128,1]
        # array of '0' or '1' denoted if the instance is solved with specified time limit
        raw_label.append(dic['index'])
        raw_runtime.append(dic['runtime'])
    print(len(raw_feature), len(raw_label), len(raw_runtime))
    return shuffle(raw_feature, raw_label, raw_runtime)
    # return [raw_feature[:1152], raw_label[:1152], raw_runtime[:1152]]


def input_data():
    """
    return: raw_feature, raw_label, raw_runtime
    """
    # tf data
    raw_feature, raw_label, raw_runtime = normal_data()
    # leng = len(raw_feature)
    leng = 1024
    arfea = np.asarray(raw_feature[:leng], dtype=np.float32).reshape(
        [leng, IMG_SIZE[0] ** 2])
    arlab = np.asarray(raw_label[:leng], dtype=np.float32)
    arrt = np.asarray(raw_runtime[:leng], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((arfea, arlab, arrt))
    iterator = ds.batch(TRAIN_BATCH_SIZE).make_initializable_iterator()
    eval_data = zip([raw_feature[leng:], raw_label[leng:], raw_runtime[leng:]])
    return (iterator, eval_data)


def get_graph(inputs):
    conv1 = conv_layer(inputs, [3, 3, 1, 32], name="Conv_1")
    conv2 = conv_layer(conv1, [2, 2, 32, 64], keep_rate=0.8, name="Conv_2")
    conv3 = conv_layer(conv2, [2, 2, 64, 128], keep_rate=0.7, name='Conv_3')
    conv_flat = tf.reshape(conv3, [-1, 16 * 16 * 128])
    with tf.variable_scope('FC_1'):
        fc1 = fc_layer(conv_flat, 16 * 16 * 128, 1000, name='FullConnect')
        drop1 = tf.nn.dropout(fc1, keep_prob=0.5, name='Dropout')
    with tf.variable_scope('FC_2'):
        fc2 = fc_layer(drop1, 1000, 200, name='FullConnect')
    output = fc_layer(fc2, 200, NUM_LABEL, tf.nn.sigmoid, name='Output')
    return output

# --------evaluation function--------------


def PAR10(runtime, index, batch, solvers):
    """
    runtime: shape of [batch, num_of_solvers] -> [batch, num, 1]
    index: shape of [batch, 1] -> [batch, solvers, 1]
    """
    onehot = tf.one_hot(
        indices=tf.cast(index, tf.int32), depth=solvers)  # PARA
    onehot_ = tf.reshape(onehot, shape=(batch, solvers, 1))
    runtime_ = tf.reshape(runtime, shape=(batch, solvers, 1))
    pre_time = tf.multiply(onehot_, tf.cast(runtime_, tf.float32))
    # mean_time = tf.reduce_mean(tf.reduce_sum(pre_time, axis=1), axis=0)
    mean_time, update_op = tf.metrics.mean(tf.reduce_sum(pre_time, axis=1))
    # return tf.reduce_sum(mean_time)
    return (mean_time, update_op)


def Mis(labels, predictions, threshold, batch):
    fn, update_op_fn = tf.metrics.false_negatives_at_thresholds(
        labels=labels, predictions=predictions, thresholds=threshold)
    fp, update_op_fp = tf.metrics.false_positives_at_thresholds(
        labels=labels, predictions=predictions, thresholds=threshold)
    fn_ = tf.reduce_sum(fn)
    fp_ = tf.reduce_sum(fp)
    return (tf.divide(tf.cast(tf.add(fn_, fp_), tf.float32), batch),
            tf.group(update_op_fn, update_op_fp))


def Percentage(labels, predictions, batch):
    onehot_pre = tf.one_hot(
        indices=tf.argmax(input=predictions, axis=1, output_type=tf.int32),
        axis=1,
        depth=PARA['num_of_solvers'])
    tp, update_op = tf.metrics.true_positives(
        labels=labels, predictions=onehot_pre)
    return (tf.divide(tp, batch) * 100, update_op)
# -----------------------------------------


def main():
    # with tf.device('/cpu:0'):
    # with tf.device('/device:GPU:0'):

    with ops.Graph().as_default() as g:
        with g.device('/CPU:0'):
            iterator, eval_data = input_data()
            fea, lab, rt = iterator.get_next()
            eva_fea, eva_lab, eva_rt = eval_data
            global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        with g.device("/device:GPU:0"):
            output = get_graph(tf.reshape(
                fea, [TRAIN_BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1]))
            loss = tf.losses.log_loss(labels=lab, predictions=output, epsilon=0)
            optimizer = tf.train.MomentumOptimizer(
                LEARNING_RATE, MOMENTUM, use_nesterov=True)

            with g.device("/CPU:0"):
                tf.summary.scalar('train_loss', loss)

            train_op = optimizer.minimize(
                loss=loss, global_step=global_step_tensor)

            # evaluating performance
            eva_out = get_graph(eva_fea)

            par10, upop1 = PAR10(eva_rt, tf.argmax(input=eva_out, axis=1),
                                 EVAL_BATCH_SIZE, NUM_LABEL)
            mis, upop2 = Mis(eva_lab, eva_out, [0.5], EVAL_BATCH_SIZE)
            per, upop3 = Percentage(eva_lab, eva_out, EVAL_BATCH_SIZE)

        # start session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = "BFC"

        # with g.device('/CPU:0'):
        sum_merge = tf.summary.merge_all()
        gs_op = tf.train.get_global_step()
        saver = tf.train.Saver()
        wrietr = tf.summary.FileWriter(TF_MODEL_PATH)

        with tf.Session(config=config, graph=g) as sess:
            wrietr.add_graph(sess.graph)
            tf.train.write_graph(sess.graph, TF_MODEL_PATH, 'graph.pbtxt')

            # restore model
            ckpt_status = tf.train.get_checkpoint_state(TF_MODEL_PATH)
            latest_ckpt = tf.train.latest_checkpoint(TF_MODEL_PATH)
            if ckpt_status and latest_ckpt:
                print('Restoreing model from {}'.format(TF_MODEL_PATH))
                saver.restore(sess, latest_ckpt)
                # gsv = int(latest_ckpt.rpartition('-')[-1])
                # gst_update = global_step_tensor.assign(gsv)
                # sess.run(gst_update)
            else:
                print('Initializing variable')
                sess.run(tf.global_variables_initializer())

            # start training
            for e in range(NUM_EPOCH):
                sess.run(iterator.initializer)
                for b in range(NUM_BATCH_OF_EPOCHS):
                    _, gs, loss_value, sm = sess.run(
                        [train_op, gs_op, loss, sum_merge])

                    vpar10, vmis, vper = sess.run([par10, mis, per])

                    wrietr.add_summary(sm, gs)
                    tf.logging.info(
                        "Loss: {}, Steps: {}".format(loss_value, gs))
                    tf.logging.info(
                        "PAR10: {}, Misclassified: {}, Percentage: {}".format(vpar10, vmis, vper))

                    if e == NUM_EPOCH - 1 and b == NUM_BATCH_OF_EPOCHS - 1:
                        model_name = '{}model.ckpt'.format(TF_MODEL_PATH)
                        saver.save(sess, model_name, global_step=gs)


if __name__ == '__main__':
    main()
