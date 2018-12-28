""" cnn model based on SAT problem """
# pep8: disable=C0103,C0111,E303
import sys
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.utils import shuffle

tf.logging.set_verbosity(tf.logging.INFO)
if sys.platform == 'linux':
    PATH_ROOT = "/home/song/Data/"
else:
    PATH_ROOT = "D:/"

DATASET = PATH_ROOT + 'SAT-INSTANCE/SAT12-RAND-data'
TF_MODEL_PATH = PATH_ROOT + 'tmp_model/SAT12-RAND/'
LOGFILE = PATH_ROOT + 'tmp_model/log/SAT12-RAND.log'

THRESHOLD = [0.5]
BATCH_SIZE_OF_TRAIN = 128
BATCH_SIZE_OF_EVAL = 82
LEARNING_RATE = 0.08
MOMENTUM = 0.96
NUM_LABEL = 31
NUM_EPOCH = 200
NUM_BATCH_OF_EPOCHS = 10
IMG_SIZE = [128, 128]


# TODO PARA


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


def cnn_model(features, labels, mode, params):
    """ building a concrete cnn model function """

    # get a input layer by features and labels, its size is 128*128*1

    with tf.device('/GPU:0'):
        input_layer = tf.reshape(
            tf.cast(features['fea'], dtype=tf.float32), [-1, 128, 128, 1])

        # get graph
        conv1 = conv_layer(input_layer, [3, 3, 1, 32], name="Conv_1")
        conv2 = conv_layer(conv1, [2, 2, 32, 64], keep_rate=0.8, name="Conv_2")
        conv3 = conv_layer(conv2, [2, 2, 64, 128],
                           keep_rate=0.7, name='Conv_3')
        conv_flat = tf.reshape(conv3, [-1, 16 * 16 * 128])
        with tf.variable_scope('FC_1'):
            fc1 = fc_layer(conv_flat, 16 * 16 * 128, 1000, name='FullConnect')
            drop1 = tf.nn.dropout(fc1, keep_prob=0.5, name='Dropout')
        with tf.variable_scope('FC_2'):
            fc2 = fc_layer(drop1, 1000, 200, name='FullConnect')
        output = fc_layer(fc2, 200, NUM_LABEL, tf.nn.sigmoid, name='Output')
    # the shape of output is [batch_size, num_labels]

    prediction = {
        'index': tf.argmax(input=output, axis=1),
        'probability': tf.nn.softmax(logits=output, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)

    # labels : the shape of [-1, num_solver]
    # labels must be multi-hot
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.log_loss(labels=labels, predictions=output, epsilon=1e-8)

    # change learning rate and momentum
    init_lr = LEARNING_RATE
    final_lr = 0.001
    init_mm = MOMENTUM
    final_mm = 0.999
    lr_step = 0.003
    mm_step = 0.001


    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.device('/GPU:0'):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=LEARNING_RATE, momentum=MOMENTUM, use_nesterov=True)
            train_op = optimizer.minimize(
                loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # custom PAR10, Percentage solved and Misclassified solvers
    # PAR10 pre-result=the time of solver corresponding to selected index which has the most high value
    # last output is mean run time of evaluation data set

    eval_metric_ops = {
        "PAR10":
            PAR10(labels, features['runtime'], prediction['index'],
                  params['batch_size_of_eval'], params['num_label']),
        "Misclassified_solver":
            Mis(labels, output, THRESHOLD, params['batch_size_of_eval']),
        "Percentage_solverd":
            Percentage(labels, output,
                       params['batch_size_of_eval'], params['num_label'])
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


def PAR10(label, runtime, index, batch, solvers):
    """
    runtime: shape of [batch, num_of_solvers] -> [batch, num, 1]
    index: shape of [batch, 1] -> [batch, solvers, 1]
    """
    onehot = tf.one_hot(
        indices=tf.cast(index, tf.int32), depth=solvers)  # PARA
    onehot_ = tf.reshape(onehot, shape=(batch, solvers, 1))
    runtime_ = tf.reshape(runtime, shape=(batch, solvers, 1))

    penalized_label = -9 * label + 10
    re_penalized_label = tf.reshape(penalized_label, shape=(batch, solvers, 1))
    penalized_pre = tf.multiply(
        onehot_, tf.cast(re_penalized_label, tf.float32))

    pre_time = tf.multiply(penalized_pre, tf.cast(runtime_, tf.float32))
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


def Percentage(labels, predictions, batch, solvers):
    onehot_pre = tf.one_hot(
        indices=tf.argmax(input=predictions, axis=1, output_type=tf.int32),
        axis=1,
        depth=solvers)
    tp, update_op = tf.metrics.true_positives(
        labels=labels, predictions=onehot_pre)
    return (tf.divide(tp, batch) * 100, update_op)


def normal_data():
    filename = DATASET
    data = joblib.load(filename)
    raw_feature = []
    raw_label = []
    raw_runtime = []
    for i, dic in data:
        # instance array image shape of [128,128,1]
        raw_feature.append(dic['aim'])
        # array of '0' or '1' denoted if the instance is solved with specified time limit
        raw_label.append(dic['index'])
        raw_runtime.append(dic['runtime'])
    print(len(raw_feature), len(raw_label), len(raw_runtime))
    return shuffle(np.asarray(raw_feature, dtype=np.float32), np.asarray(raw_label, dtype=np.float32),
                   np.asarray(raw_runtime, dtype=np.float32))
    # return [raw_feature[:1152], raw_label[:1152], raw_runtime[:1152]]


def input_data(shuffle=False):
    """
    return: raw_feature, raw_label, raw_runtime
    """
    # tf data
    raw_feature, raw_label, raw_runtime = normal_data()
    # leng = len(raw_feature)
    intercept = NUM_BATCH_OF_EPOCHS * BATCH_SIZE_OF_TRAIN
    # arfea = np.asarray(raw_feature[:leng], dtype=np.float32).reshape(
    #     [leng, IMG_SIZE[0] ** 2])
    # arlab = np.asarray(raw_label[:leng], dtype=np.float32)
    # arrt = np.asarray(raw_runtime[:leng], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices(
        (raw_feature[:intercept], raw_label[:intercept], raw_runtime[:intercept]))
    # iterator = ds.batch(BATCH_SIZE_OF_TRAIN).make_initializable_iterator()
    eval_data = (raw_feature[intercept:],
                 raw_label[intercept:], raw_runtime[intercept:])

    def my_input_fn(ds=ds, shuffle=shuffle):
        if shuffle:
            ds = ds.shuffle(1000)
        iterator = ds.repeat(NUM_EPOCH).batch(
            BATCH_SIZE_OF_TRAIN).make_one_shot_iterator()
        feature, label, runtime = iterator.get_next()
        features = {'fea': feature}
        return (features, label)

    return (my_input_fn, eval_data)


def main():
    # get data
    input_fn, eval_data = input_data()
    eval_fea, eval_lab, eval_rt = eval_data
    PARA = dict({'batch_size': BATCH_SIZE_OF_TRAIN,
                 'num_label': NUM_LABEL, 'batch_size_of_eval': BATCH_SIZE_OF_EVAL})

    runConf = tf.estimator.RunConfig(
        save_summary_steps=50, log_step_count_steps=100)

    # create the estimator
    sat_clf = tf.estimator.Estimator(
        model_fn=cnn_model,
        model_dir=TF_MODEL_PATH,
        params=PARA,
        config=runConf)

    # Set up logging for predictions
    # steps = tf.train.get_global_step(ops.Graph())
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # tensors_to_log = {"output": "Output"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=10)

    # def feed_fn():
    #     return dict({'fea': eval_fea, 'runtime':eval_rt})
    # feed_hook = tf.train.FeedFnHook(feed_fn=feed_fn)
    #
    # hook_sum = tf.summary.scalar()
    # summary_hook = tf.train.SummarySaverHook(save_steps=10, output_dir=TF_MODEL_PATH, summary_op=hook_sum)

    # training the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={'fea': feature},
    #     y=label,
    #     batch_size=BATCH_SIZE_OF_TRAIN,
    #     num_epochs=NUM_EPOCH,
    #     shuffle=True)
    # sat_clf.train(input_fn=input_fn, logging_hook=[logging_hook])
    sat_clf.train(input_fn=input_fn)

    # return
    # evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'fea': eval_fea,
           'runtime': eval_rt},
        y=eval_lab,
        batch_size=BATCH_SIZE_OF_EVAL,
        num_epochs=1,
        shuffle=False)
    eval_res = sat_clf.evaluate(input_fn=eval_input_fn)
    # logger = ins_log()
    # logger.info(eval_res)
    print(eval_res)

    # TODO prediction test


# def ins_log(level=logging.INFO):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(level)
#
#     han = logging.StreamHandler(open(LOGFILE, 'a'))
#     han.setLevel(level)
#     formatter = logging.Formatter(
#         "%(asctime)s @%(name)s #%(levelname)s : %(message)s")
#     han.setFormatter(formatter)
#     logger.addHandler(han)
#     return logger


if __name__ == '__main__':
    main()
