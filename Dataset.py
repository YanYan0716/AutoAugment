import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import pandas as pd
import random
from AutoAugment import cutout, apply_policy


import config


class DataAugment(object):
    def __init__(self, auto_augment=False, cutout=True):
        self.auto_augment = auto_augment
        self.cutout = cutout
        if self.auto_augment:
            self.policies = [
                ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
                ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
                ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
                ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
                ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
                ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
                ['Color', 0.4, 3, 'Brightness', 0.6, 7],
                ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
                ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
                ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
                ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
                ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
                ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
                ['Brightness', 0.9, 6, 'Color', 0.2, 8],
                ['Solarize', 0.5, 2, 'Invert', 0, 0.3],
                ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
                ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
                ['Color', 0.9, 9, 'Equalize', 0.6, 6],
                ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
                ['Brightness', 0.1, 3, 'Color', 0.7, 0],
                ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
                ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
                ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
                ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
                ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
            ]

    def augment(self, img):
        if self.cutout:
            img = tf.numpy_function(cutout, [img], tf.float32)
        if self.auto_augment:
            img = img.astype('uint8')
            img = apply_policy(img, self.policies[random.randrange(len(self.policies))])

        return img


# 制作有标签的数据集
def label_image(img_file, label):
    '''
    获取图片，对图片做水平翻转 随机剪裁等， label变为onehot
    :param img_file:
    :param label:
    :return:
    '''
    # 对图片的处理
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, (config.IMG_SIZE + 5, config.IMG_SIZE + 5))
    img = tf.image.random_crop(img, (config.IMG_SIZE, config.IMG_SIZE, 3))
    # img = img.numpy()
    aug = DataAugment()
    img = aug.augment(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    img = tf.cast(img, config.DTYPE) / 255.0
    mean = tf.expand_dims(tf.convert_to_tensor([0.4914, 0.4822, 0.4465], dtype=config.DTYPE), axis=0)
    std = tf.expand_dims(tf.convert_to_tensor([0.2471, 0.2435, 0.2616], dtype=config.DTYPE), axis=0)
    img = (img-mean)/std+(1e-6)

    # 对标签的处理
    label = tf.raw_ops.OneHot(indices=label, depth=config.NUM_CLASS, on_value=1.0, off_value=0)
    label = tf.cast(label, dtype=config.DTYPE)
    return {'images': img, 'labels': label}


if __name__ == '__main__':

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # 有标签的数据集 batch_size=config.BATCH_SIZE
    df_label = pd.read_csv('./label4000.csv')
    file_paths = df_label['file_name'].values
    labels = df_label['label'].values
    ds_label_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_label_train = ds_label_train\
        .map(label_image, num_parallel_calls=AUTOTUNE)\
        .batch(1).shuffle(1)
    for data in ds_label_train:
        print(data.keys())
        break

