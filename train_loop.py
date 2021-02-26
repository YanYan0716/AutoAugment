import tensorflow as tf

import config

def train_loop(dataset, model, coslr, global_step, optimizer, Loss, acc_metric):
    loss_mean = 0
    for batch_idx, (img, label) in enumerate(dataset):
        with tf.GradientTape() as tape:
            y_pred = model(img, training=True)
            loss = Loss(label, y_pred)
            loss_mean += loss
        grad = tape.gradient(loss, model.trainable_weights)
        lr = coslr.__call__(global_step)
        optimizer(lr, config.MOMENTUM).apply_gradients(zip(grad, model.trainable_weights))
        acc_metric.update_state(y_pred, label)
    train_acc = acc_metric.result()
    loss_mean = loss_mean / batch_idx
    acc_metric.reset_states()
    return model, loss_mean, train_acc


def test_loop(dataset, model, acc_metric):
    for batch_idx, (img, label) in enumerate(dataset):
        y_pred = model(img, training=False)
        acc_metric.update_state(label, y_pred)
    test_acc = acc_metric.result()
    acc_metric.reset_states()
    return test_acc