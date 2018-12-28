""" cnn model based on SAT problem """
# pylint: disable=C0103,C0111
import numpy as np
import tensorflow as tf
import datetime as dt
import logging
from sklearn.externals import joblib
from sklearn.utils import shuffle

tf.logging.set_verbosity(tf.logging.INFO)
PARA = {'batch_size': 128, 'num_of_solvers': 31, 'num_of_eval_data': 128}
THRESHOLD = [0.5]
DATASET = 'D:/SAT-instance/SAT12-indu-data'
LOGFILE = 'D:/SAT-instance/log/SAT12-indu.txt'


def cnn_model(features, labels, mode, params):
    """ building a concrete cnn model function """

    # get a input layer by features and labels, its size is 128*128*1

    input_layer = tf.reshape(
        tf.cast(features['aim'], dtype=tf.float32), [-1, 128, 128, 1])

    # first convolutional layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu)
    pooling1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[2, 2], strides=2)
    dropout1 = tf.layers.dropout(inputs=pooling1, rate=0.1)

    # second convolutional layer
    conv2 = tf.layers.conv2d(
        inputs=dropout1,
        filters=64,
        kernel_size=[2, 2],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu)
    pooling2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[2, 2], strides=2)
    dropout2 = tf.layers.dropout(inputs=pooling2, rate=0.2)

    # third convolutinal layer
    conv3 = tf.layers.conv2d(
        inputs=dropout2,
        filters=128,
        kernel_size=[2, 2],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu)
    pooling3 = tf.layers.max_pooling2d(
        inputs=conv3, pool_size=[2, 2], strides=2)
    dropout3 = tf.layers.dropout(inputs=pooling3, rate=0.3)

    # fully connected layer
    dropout3_flat = tf.reshape(dropout3, [-1, 16 * 16 * 128])
    dense1 = tf.layers.dense(
        inputs=dropout3_flat, units=1000, activation=tf.nn.relu)
    dropout4 = tf.layers.dropout(inputs=dense1, rate=0.5)
    dense2 = tf.layers.dense(inputs=dropout4, units=200, activation=tf.nn.relu)
    output = tf.layers.dense(
        inputs=dense2,
        units=params['num_of_solvers'],
        activation=tf.nn.sigmoid)  # PARA
    # the shape of output is [batch_size, num_labels, 1]

    prediction = {
        'index': tf.argmax(input=output, axis=1),
        'probability': tf.nn.softmax(logits=output, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)

    # labels : the shape of [-1, num_solver]
    # labels must be multi-hot
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.log_loss(labels=labels, predictions=output, epsilon=1e-7)
    # loss_sum = tf.summary.scalar(name='loss_sum', tensor=loss)
    # loss_his = tf.summary.histogram(name='loss_his', values=loss)
    # summary_hook = tf.train.SummarySaverHook(
    #     save_steps=10,
    #     output_dir='D:/training_model/sat_summary_tra/',
    #     summary_op=tf.summary.merge_all())
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.3, momentum=0.9, use_nesterov=True)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # custome PAR10, Percentage solved and Misclassified solvers
    # PAR10 pre-result=the time of solver corresponding to selected index which has the most high value
    # last output is mean run time of evaluation data set

    eval_metric_ops = {
        "PAR10":
        PAR10(features['runtime'], prediction['index'],
              params['num_of_eval_data'], params['num_of_solvers']),
        "Misclassified_solver":
        Mis(labels, output, THRESHOLD, params['num_of_eval_data']),
        "Percentage_solverd":
        Percentage(labels, output, params['num_of_eval_data'])
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


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


# def main():
#     # get data
#     len_tra = 1024
#     raw_data, raw_label, raw_runtime = normal_data()
#     shuffle(raw_data, raw_label, raw_runtime)
#     training_data = np.asarray(raw_data[0:len_tra])
#     training_label = np.asarray(raw_label[0:len_tra])
#     eval_data = np.asarray(raw_data[len_tra:-1])
#     eval_label = np.asarray(raw_label[len_tra:-1])
#     eval_runtime = np.asarray(raw_runtime[len_tra:-1])

#     # create the estimator
#     sat_clf = tf.estimator.Estimator(
#         model_fn=cnn_model,
#         model_dir='D:/training_model/sat12_indu_cnn',
#         params=PARA)

#     # Set up logging for predictions
#     # steps = tf.train.get_global_step(graph=)
#     tensors_to_log = {"probabilities": "softmax_tensor"}
#     # tensors_to_log = {"accuracy": "accuracy_tensor:0"}
#     logging_hook = tf.train.LoggingTensorHook(
#         tensors=tensors_to_log, every_n_iter=10)

#     # training the model
#     # train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     #     x={'aim': training_data},
#     #     y=training_label,
#     #     batch_size=128,
#     #     num_epochs=100,
#     #     shuffle=True)
#     # sat_clf.train(input_fn=train_input_fn, hooks=[logging_hook])

#     # evaluate the model
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={'aim': eval_data,
#            'runtime': eval_runtime},
#         y=eval_label,
#         batch_size=129,
#         num_epochs=1,
#         shuffle=False)
#     eval_res = sat_clf.evaluate(input_fn=eval_input_fn)
#     logger = ins_log()
#     logger.info(eval_res)
#     print(eval_res)

#     # TODO prediction test


def normal_data():
    filename = DATASET
    data = joblib.load(filename)
    raw_data = []
    raw_label = []
    raw_runtime = []
    for i, dic in data:
        raw_data.append(dic['aim'])
        label = dic['index']
        # label.resize(len(label), 1)
        raw_label.append(label)
        runtime = raw_runtime.append(dic['runtime'])
    print(len(raw_data), len(raw_label), len(raw_runtime))
    return (raw_data, raw_label, raw_runtime)


def ins_log(level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    han = logging.StreamHandler(open(LOGFILE, 'a'))
    han.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s @%(name)s #%(levelname)s : %(message)s")
    han.setFormatter(formatter)
    logger.addHandler(han)
    return logger


def cv(training_data, training_label, eval_data, eval_label, eval_runtime,
       name):
    TRAINING_BATCH = PARA['batch_size']
    EVAL_BATCH = PARA['num_of_eval_data']

    # create the estimator
    sat_clf = tf.estimator.Estimator(
        model_fn=cnn_model,
        model_dir='D:/training_model/sat12_indu_cnn' + name,
        params=PARA)

    # Set up logging for predictions
    # steps = tf.train.get_global_step(graph=)
    tensors_to_log = {"probabilities": "softmax_tensor"}
    # tensors_to_log = {"accuracy": "accuracy_tensor:0"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    # training the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'aim': training_data},
        y=training_label,
        batch_size=TRAINING_BATCH,
        num_epochs=200,
        shuffle=True)
    sat_clf.train(input_fn=train_input_fn, hooks=[logging_hook])

    # evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'aim': eval_data,
           'runtime': eval_runtime},
        y=eval_label,
        batch_size=EVAL_BATCH,
        num_epochs=1,
        shuffle=False)
    eval_res = sat_clf.evaluate(input_fn=eval_input_fn)
    logger = ins_log()
    logger.info(eval_res)
    print(eval_res)


def main():
    # len_tra = 1024
    raw_data, raw_label, raw_runtime = normal_data()
    shuffle(raw_data, raw_label, raw_runtime)
    # training_data = np.asarray(raw_data[0:len_tra])
    # training_label = np.asarray(raw_label[0:len_tra])
    # eval_data = np.asarray(raw_data[len_tra:-1])
    # eval_label = np.asarray(raw_label[len_tra:-1])
    # eval_runtime = np.asarray(raw_runtime[len_tra:-1])
    TRAIN_BATCH = PARA['batch_size']
    EVAL_BATCH = PARA['num_of_eval_data']
    for i in range(9):
        eval_data = np.asarray(raw_data[i * TRAIN_BATCH:(i + 1) * TRAIN_BATCH])
        eval_label = np.asarray(raw_label[i * TRAIN_BATCH:(i + 1) * TRAIN_BATCH])
        eval_runtime = np.asarray(raw_runtime[i * TRAIN_BATCH:(i + 1) * TRAIN_BATCH])

        tmp1 = raw_data[0:i * TRAIN_BATCH]
        tmp2 = raw_data[(i + 1) * TRAIN_BATCH:-2]
        if tmp1!=[] and tmp2!=[]:
            training_data = np.concatenate((tmp1, tmp2), axis=0)
        else:
            if tmp1 == []:
                training_data = np.asarray(tmp2)
            else:
                training_data = np.asarray(tmp1)
        tmp1 = raw_label[0:i * TRAIN_BATCH]
        tmp2 = raw_label[(i + 1) * TRAIN_BATCH:-2]
        if tmp1 != [] and tmp2 != []:
            training_label = np.concatenate((tmp1, tmp2), axis=0)
        else:
            if tmp1 == []:
                training_label = np.asarray(tmp2)
            else:
                training_label = np.asarray(tmp1)
        name = '_cv' + str(i)
        cv(training_data, training_label, eval_data, eval_label, eval_runtime,
           name)


if __name__ == '__main__':
    main()