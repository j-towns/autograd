"""Some standard gradient-based stochastic optimizers.

These are just standard routines that don't make any use of autograd,
though you could take gradients of these functions too if you want
to do meta-optimization.

These routines can optimize functions whose inputs are structured
objects, such as dicts of numpy arrays."""
from __future__ import absolute_import
from builtins import range

import autograd.numpy as np
from autograd.misc import flatten
from autograd.wrap_util import wraps

def unflatten_optimizer(optimize):
    """Takes an optimizer that operates on flat 1D numpy arrays and returns a
    wrapped version that handles trees of nested containers (lists/tuples/dicts)
    with arrays/scalars at the leaves."""
    @wraps(optimize)
    def _optimize(val_and_grad, x0, callback=None, *args, **kwargs):
        _x0, unflatten = flatten(x0)
        def _val_and_grad(x, i):
            val, grad = val_and_grad(unflatten(x), i)
            return val, flatten(grad)[0]
        if callback:
            _callback = lambda x, i, g, v: callback(unflatten(x), i, unflatten(g), v)
        else:
            _callback = None
        return unflatten(optimize(_val_and_grad, _x0, _callback, *args, **kwargs))

    return _optimize

@unflatten_optimizer
def adam(val_and_grad, x, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        val, g = val_and_grad(x, i)
        if callback: callback(x, i, g, val)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return x
