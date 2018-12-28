import os
import math
import numpy as np
import tensorflow as tf
from PIL import Image

# image name suffix format: _ + resample method + size +
# vector convert method([PIL]: directly using PIL.Image model;[NUM]: reshape using numpy model)
# resample method: [NEAREST, BILINEAR, BICUBIC, LANCZOS]
IMG_SUFFIX = '_LAN128NUM'  # first some letter is resample method


# convertion with less memory


def convert_instace(ins_filename, resample=Image.LANCZOS, size=[128, 128]):
    # ins_filename include path
    # print(ins_filename)
    num_file = []
    name = ins_filename[:ins_filename.rindex('.')] + IMG_SUFFIX  # /path/name (without type suffix)
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
            tem = [ord(_) for _ in lines[0:lines.rindex('0')-1]]
            tem.append(0)
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
    im.save('{}.bmp'.format(name), format='BMP')
    im.close()
