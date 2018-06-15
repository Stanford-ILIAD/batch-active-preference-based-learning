import numpy as np
import utils_driving as utils
import theano as th
import theano.tensor as tt
from trajectory import Trajectory
import feature

class Car(object):
    def __init__(self, dyn, x0, color='yellow', T=5):
        self.data0 = {'x0': x0}
        self.bounds = [(-1., 1.), (-1., 1.)]
        self.T = T
        self.dyn = dyn
        self.traj = Trajectory(T, dyn)
        self.traj.x0.set_value(x0)
        self.linear = Trajectory(T, dyn)
        self.linear.x0.set_value(x0)
        self.color = color
        self.default_u = np.zeros(self.dyn.nu)
    def reset(self):
        self.traj.x0.set_value(self.data0['x0'])
        self.linear.x0.set_value(self.data0['x0'])
        for t in range(self.T):
            self.traj.u[t].set_value(np.zeros(self.dyn.nu))
            self.linear.u[t].set_value(self.default_u)
    def move(self):
        self.traj.tick()
        self.linear.x0.set_value(self.traj.x0.get_value())
    @property
    def x(self):
        return self.traj.x0.get_value()
    @x.setter
    def x(self, value):
        self.traj.x0.set_value(value)
    @property
    def u(self):
        return self.traj.u[0].get_value()
    @u.setter
    def u(self, value):
        self.traj.u[0].set_value(value)
    def control(self, steer, gas):
        pass
