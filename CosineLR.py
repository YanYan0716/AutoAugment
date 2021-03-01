import math
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K


class CosineLR():
    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineLR, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def __call__(self, global_step, **kwargs):
        lr = self.eta_min + \
             (self.eta_max - self.eta_min)*(1+math.cos(math.pi*global_step/self.T_max))/2
        return lr


class CosineAnnealingScheduler(callbacks.Callback):
    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('no _lr_ atribute.')
        lr = self.eta_min+(self.eta_max-self.eta_min)*(1+math.cos(math.pi*epoch/self.T_max))/2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))