import theano as th
import theano.tensor as tt
import utils_driving as utils
import numpy as np

class Trajectory(object):
    def __init__(self, T, dyn):
        self.dyn = dyn
        self.T = T
        self.x0 = utils.vector(dyn.nx)
        self.u = [utils.vector(dyn.nu) for t in range(self.T)]
        self.x = []
        z = self.x0
        for t in range(T):
            z = dyn(z, self.u[t])
            self.x.append(z)
        self.next_x = th.function([], self.x[0])
    def tick(self):
        self.x0.set_value(self.next_x())
        for t in range(self.T-1):
            self.u[t].set_value(self.u[t+1].get_value())
        self.u[self.T-1].set_value(np.zeros(self.dyn.nu))
