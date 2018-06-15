import theano as th
import theano.tensor as tt
import numpy as np

def extract(var):
    return th.function([], var, mode=th.compile.Mode(linker='py'))()

def shape(var):
    return extract(var.shape)

def vector(n):
    return th.shared(np.zeros(n))

def matrix(n, m):
    return tt.shared(np.zeros((n, m)))

def grad(f, x, constants=[]):
    ret = th.gradient.grad(f, x, consider_constant=constants, disconnected_inputs='warn')
    if isinstance(ret, list):
        ret = tt.concatenate(ret)
    return ret

def jacobian(f, x, constants=[]):
    sz = shape(f)
    return tt.stacklists([grad(f[i], x) for i in range(sz)])
    ret = th.gradient.jacobian(f, x, consider_constant=constants)
    if isinstance(ret, list):
        ret = tt.concatenate(ret, axis=1)
    return ret

def hessian(f, x, constants=[]):
    return jacobian(grad(f, x, constants=constants), x, constants=constants)

