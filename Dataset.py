import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import pandas as pd
from AutoAugment import tfcutout, tfapply_policy
import matplotlib.pyplot as plt


import config


# 数据集
def train_aug(img_file, label):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, (config.IMG_SIZE + 5, config.IMG_SIZE + 5))
    img = tf.image.random_crop(img, (config.IMG_SIZE, config.IMG_SIZE, 3))
    if config.CUTOUT:
        img = tfcutout(img)
    if config.AUTO_AUGMENT:
        img = tfapply_policy(img)

    img = tf.cast(img, config.DTYPE) / 255.0
    mean = tf.expand_dims(tf.convert_to_tensor([0.4914, 0.4822, 0.4465], dtype=config.DTYPE), axis=0)
    std = tf.expand_dims(tf.convert_to_tensor([0.2471, 0.2435, 0.2616], dtype=config.DTYPE), axis=0)
    img = (img-mean)/(std+(1e-6))

    # 对标签的处理
    # label = tf.raw_ops.OneHot(indices=label, depth=config.NUM_CLASS, on_value=1.0, off_value=0)
    label = tf.cast(label, dtype=config.DTYPE)
    return (img, label)


def test_aug(img_file, label):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = tf.cast(img, config.DTYPE) / 255.0
    mean = tf.expand_dims(tf.convert_to_tensor([0.4914, 0.4822, 0.4465], dtype=config.DTYPE), axis=0)
    std = tf.expand_dims(tf.convert_to_tensor([0.2471, 0.2435, 0.2616], dtype=config.DTYPE), axis=0)
    img = (img-mean)/(std+(1e-6))

    # 对标签的处理
    # label = tf.raw_ops.OneHot(indices=label, depth=config.NUM_CLASS, on_value=1.0, off_value=0)
    label = tf.cast(label, dtype=config.DTYPE)
    return (img, label)


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
        plt.imshow(data['images'][0])
        plt.axis('off')
        plt.show()
        print(data['labels'])
        break

