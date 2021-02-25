import math


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