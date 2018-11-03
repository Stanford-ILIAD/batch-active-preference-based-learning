import pymc as mc
import numpy as np
import theano as th
import theano.tensor as tt
import theano.tensor.nnet as tn
from theano.ifelse import ifelse
from scipy.stats import gaussian_kde
from utils import matrix

class Sampler(object):
    def __init__(self, D):
        self.D = D
        self.Avar = matrix(0, self.D)
        self.yvar = matrix(0, 1)
        x = tt.vector()
        self.f = th.function([x], -tt.sum(tn.relu(tt.dot(-tt.tile(self.yvar,[1,D])*self.Avar, x))))
    @property
    def A(self):
        return self.Avar.get_value()
    @A.setter
    def A(self, value):
        if len(value)==0:
            self.Avar.set_value(np.zeros((0, self.D)))
        else:
            self.Avar.set_value(np.asarray(value))
    @property
    def y(self):
        return self.yvar.get_value()
    @y.setter
    def y(self, value):
        if len(value)==0:
            self.yvar.set_value(np.zeros((0, 1)))
        else:
            self.yvar.set_value(np.asarray(value))

    def sample(self, N, T=50, burn=1000):
        x = mc.Uniform('x', -np.ones(self.D), np.ones(self.D), value=np.zeros(self.D))
        def sphere(x):
            if (x**2).sum()>=1.:
                return -np.inf
            else:
                return self.f(x)
        p1 = mc.Potential(
            logp = sphere,
            name = 'sphere',
            parents = {'x': x},
            doc = 'Sphere potential',
            verbose = 0)
        chain = mc.MCMC([x])
        chain.use_step_method(mc.AdaptiveMetropolis, x, delay=burn, cov=np.eye(self.D)/10000)
        chain.sample(N*T+burn, thin=T, burn=burn, verbose=-1)
        samples = x.trace()
        samples = np.array([x/np.linalg.norm(x) for x in samples])
        return samples
