import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd


import config
from WideResnet import WideResnet
from Dataset import train_aug, test_aug
from CosineLR import CosineLR
from train_loop import train_loop
from train_loop import test_loop


if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train dataset
    train_csv = pd.read_csv(config.TRAIN_FILE_PATH)
    file_paths = train_csv['name'].values
    labels = train_csv['label'].values
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_train = ds_train \
        .map(train_aug, num_parallel_calls=AUTOTUNE)\
        .cache() \
        .batch(config.BATCH_SIZE, drop_remainder=True) \
        .shuffle(buffer_size=50000)\
        .prefetch(AUTOTUNE)
    # test dataset
    test_csv = pd.read_csv(config.TEST_FILE_PATH)
    file_paths = test_csv['file_name'].values
    labels = test_csv['label'].values
    ds_test = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_test = ds_test \
        .map(test_aug, num_parallel_calls=AUTOTUNE)\
        .cache() \
        .batch(config.BATCH_SIZE, drop_remainder=True) \
        .shuffle(buffer_size=10000)\
        .prefetch(AUTOTUNE)

    # create model
    k = [16, 16 * config.WIDTH, 64 * config.WIDTH, 64 * config.WIDTH]
    model = WideResnet(k).model()

    # optimizer
    optimizer = keras.optimizers.SGD

    # loss
    loss_fun = keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.
    )
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    test_acc_metric = keras.metrics.CategoricalAccuracy()

    # train and test
    best_acc = 0
    print('training')
    for i in range(config.MAX_EPOCH):
        coslr = CosineLR(T_max=config.MAX_EPOCH, eta_max=config.ETA_MAX, eta_min=config.ETA_MIN)
        model, loss, train_acc = train_loop(
            dataset=ds_train,
            model=model,
            coslr=coslr,
            global_step=i,
            optimizer=optimizer,
            Loss=loss_fun,
            acc_metric=train_acc_metric,
        )
        print(f'train epoch: {i} -->  acc: {train_acc}, loss: {loss}. lr: {coslr.__call__(i)}')
        if (i+1) % config.EVA_EPOCH == 0:
            test_acc = test_loop(
                dataset=ds_test,
                model=model,
                acc_metric=test_acc_metric,
            )
            print(f'test ---------->  acc: {test_acc}')
            if test_acc > best_acc:
                model.save(config.SAVE_PATH+'_'+str(i))