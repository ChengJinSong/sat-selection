# pylint: disable=C0103,C0111
# this file deal with cnf file format convertion
# it would do that by search certain file through algorithm_run file

import logging
import math
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.externals import joblib
from sklearn import preprocessing as pp

# determine the type of system
if sys.platform == 'linux':
    path_prefix = "/home/song/Data/"
else:
    path_prefix = "D:/"

if len(sys.argv) > 1:
    data_name = sys.argv[1]
    ALGO_RUN = '{}ASLIB/{}/algorithm_runs.arff'.format(path_prefix, data_name)
    CSV_NAME = '{}SAT-INSTANCE/{}.csv'.format(path_prefix, data_name)
    TMP_DATA_NAME = '{}-data'.format(data_name)
else:
    print('Invalid data name')

size = (128, 128)
INS_FOLDER = path_prefix + 'SAT-INSTANCE/'
BIGCNF = path_prefix + "SAT-INSTANCE/big400.txt"
LOGFILE = path_prefix + "SAT-INSTANCE/log/get_image.log"
# image name suffix format: _ + resample method + size +
# vector convert method([PIL]: directly using PIL.Image model;[NUM]: reshape using numpy model)
# resample method: [NEAREST, BILINEAR, BICUBIC, LANCZOS]
IMG_SUFFIX = '_NEA128NUM'  # first some letter is resample method


def get_csv(algo_run, csv_name):
    print(algo_run)
    f = open(algo_run)
    print(f.readline())
    lines = f.readlines()
    isdata = 0
    dic = {'filename': [], 'repetition': [],
           'algorithm': [], 'runtime': [], 'runstatus': []}
    for line in lines:
        if not isdata:
            if line == '@DATA\n':
                isdata = 1
            continue
        filename, repetition, algorithm, runtime, runstatus = line.split(',')
        dic['filename'].append(filename)
        dic['repetition'].append(repetition)
        dic['algorithm'].append(algorithm)
        dic['runtime'].append(runtime)
        dic['runstatus'].append(runstatus.rstrip())
    f.close()
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(csv_name, index=False)


def im_save(im_obj, name):
    """
    name without format suffix
    """
    im_obj.save('{}.bmp'.format(name), format='BMP')


def convert_instace(ins_filename, resample=Image.NEAREST):
    # print(ins_filename)
    num_file = []
    # /path/name (without type suffix)
    name = ins_filename[:ins_filename.rindex('.')] + IMG_SUFFIX
    if os.path.isfile('{}.bmp'.format(name)):
        return
    print(name)
    with open(ins_filename) as f:
        mul_lines = f.readlines()
        last_sign = 0
        for lines in mul_lines:
            if not last_sign:
                if lines.find('p') == 0:
                    last_sign = 1
                continue
            tem = [ord(_) for _ in lines]
            num_file.extend(tem)

    # reshape by using numpy
    wh = round(math.sqrt(len(num_file)))
    difference = wh * wh - len(num_file)
    # print('diff: {}'.format(difference))
    # print('min: {}'.format(min(num_file)))
    # print('max: {}'.format(max(num_file)))
    # print('range: {}'.format(list(set(num_file))))
    if difference > 0:
        diff_zero = np.zeros(shape=(difference)).tolist()
        num_file.extend(diff_zero)
    else:
        for i in range(abs(difference)):
            del num_file[-1]
    arr = np.asarray(num_file, dtype=np.int32).reshape((wh, wh))
    # end

    # arr = np.asarray(num_file).reshape((len(num_file), 1))
    im = Image.fromarray(arr)
    im = im.convert(mode='L')
    im = im.resize(size, resample=resample)
    im_save(im, name)
    im.close()


def find_conv(in_algo, name=None):
    """
    get instance and its algorithm information from algorithm_runs file in ASLib data
    then find one instance and convert it to image, store it
    """
    f = open(BIGCNF, 'r')
    fn = f.readlines()
    f.close()
    bigfn = [
        "{}{}".format(INS_FOLDER, n.strip('./').strip('\n')) for n in fn
    ]
    del_index = []
    loger = ins_log()
    for i in range(0, len(in_algo)):
        print(i)
        filename = in_algo[i][0]
        if not os.path.isfile(filename):
            print('Can not find file: {}'.format(filename))
            loger.error('Can not find file: {}'.format(filename))
            continue
        # dealing big cnf file >400M
        # print(filename)
        if filename in bigfn:
            del_index.append(i)
            continue
        # end_____

        if not os.path.isfile('{}{}.bmp'.format(
                filename[:filename.rindex('.')], IMG_SUFFIX)):
            convert_instace(filename)
        image_ = "{}{}.bmp".format(filename[:filename.rindex('.')], IMG_SUFFIX)
        im = Image.open(image_)
        aim = np.asarray(im)
        in_algo[i][1]['aim'] = aim

    tmp = 0
    for ii in del_index:
        del in_algo[ii - tmp]
        tmp += 1
    # SD standardization
    in_algo_ = normalization(in_algo)
    joblib.dump(in_algo_, '{}{}'.format(INS_FOLDER, name))


def normalization(in_algo):
    """
    preprocessed by subtracting the mean and normalizing each feature
    """
    imarr = []
    for i in range(0, len(in_algo)):
        imarr.extend(in_algo[i][1]['aim'].tolist())
    # scaler = pp.StandardScaler()
    # scaler.fit(imarr)
    # scaler.transform(imarr)
    scaled = pp.scale(imarr)
    for i in range(0, len(in_algo)):
        in_algo[i][1]['aim'] = np.asarray(
            scaled[i * 128:(i + 1) * 128]).reshape(128, 128, 1)
    return in_algo


def get_data(algo_run, csv_name):
    get_csv(algo_run, csv_name)
    df = pd.read_csv(csv_name)
    instance_list = df['filename'].drop_duplicates().values
    in_algo = []
    for name_ in instance_list:
        instance_df = df[df.filename == name_].sort_values(by='algorithm')
        instance_df = instance_df.drop(columns='filename')
        instance_dic = instance_df.to_dict(orient='list')
        instance_dic = addIndex(instance_dic)
        # instance_dic['index'] = np.asarray(instance_dic['index'])
        instance_dic['runtime'] = np.asarray(
            instance_dic['runtime'], dtype=np.float)
        instance_dic['runstatus'] = np.asarray(instance_dic['runstatus'])
        if name_[0] == '.':
            name_ = name_[2:]
        name = '{}{}'.format(INS_FOLDER, name_)
        in_algo.append([name, instance_dic])
    return in_algo


def addIndex(instance_dic):
    """
    instance_dic : only have one instance infomation
    setting label of index of run status, ok or timeout
    """
    index = []
    for sta in instance_dic['runstatus']:
        if sta == 'ok':
            sign = 1
        else:
            sign = 0
        index.append(sign)
    instance_dic['index'] = np.asarray(index)
    return instance_dic


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


find_conv(get_data(ALGO_RUN, CSV_NAME), name=TMP_DATA_NAME)
# da = joblib.load('D:/SAT-instance/SAT11-indu-data')
# normalization(da)
