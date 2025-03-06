import jax.numpy as jnp

def steps(high=0.1, nhigh=200, small=0.01, nsmall=450, vsmall=1e-3):
    def inner_step(iter_num):
        return jnp.where(iter_num < nhigh, high,
                         jnp.where(iter_num < nsmall+nhigh, small, vsmall))
    return inner_step
                         
