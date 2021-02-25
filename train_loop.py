import tensorflow as tf

from CosineLR import CosineLR


def train_loop(dataset, model, coslr, global_step, optimizer, Loss):
    for batch_idx, (img, label) in enumerate(dataset):
        with tf.GradientTape() as tape:
            y_pred = model(img, training=True)
            loss = Loss(label, y_pred)

        grad = tape.gradient(loss, model.trainable_weights)
        lr = coslr.__call__(global_step)
        optimizer(lr=lr).apply_gradients(zip(grad, model.trainable_weights))


def test_loop(dataset, model, acc_metric):
    for batch_idx, (img, label) in enumerate(dataset):
        y_pred = model(img, training=False)
        acc_metric.update_state(label, y_pred)
    test_acc = acc_metric.result()
    return test_acc