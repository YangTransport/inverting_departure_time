from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.optimize import minimize, bisect

import numpy as np

import jax
from jax import vmap
from jax.scipy.stats import norm
import jax.numpy as jnp
from jaxopt import GradientDescent
from jaxopt import Bisection

import time
from tqdm import tqdm

import generate_data
from importlib import reload
generate_data = reload(generate_data)
generate_arrival = generate_data.generate_arrival
find_td = generate_data.find_td

#%%
def asymm_gaussian(sigma_l=.7, sigma_r=.2, mu=9.5):
    left_gau = lambda x: jnp.exp(-(x-mu)**2/sigma_l**2)
    right_gau = lambda x: jnp.exp(-(x-mu)**2/sigma_r**2)
    return lambda t: jnp.where(t < 0, left_gau(t), right_gau(t))

def asymm_gaussian_plateau(sigma_l=.7, sigma_r=.2, mu=9.5, plateau_len=3):
    left_gau = lambda x: jnp.exp(-(x-mu)**2/sigma_l**2)
    right_gau = lambda x: jnp.exp(-(x-mu)**2/sigma_r**2)
    right_fun = lambda t: jnp.where(t - mu > plateau_len/2, right_gau(t - plateau_len/2), 1)
    return lambda t: jnp.where(t - mu < -plateau_len/2, left_gau(t + plateau_len/2), right_fun(t))

#%%

def likelihood_kink(travel_time, t_a, mu_b, sigma):
    """ Finds the likelihood of a point being a kink minimum,
    for the travel time, beta and gamma distributions determined by the parameters.
    Beta and gamma are assumed to be normally distributed.
    Gamma still has to be implemented
    """
    def find_bs(beta, travel_time):
        """ Given a travel time function and a beta value,
        finds the interval in which the optimal arrival time is constant
        (and there are thuss no kink minima).

        Returns a couple containing initial and final points of the interval
        """
        
        # A gradient descent algorithm finds the initial point
        in_obj = lambda x: travel_time(x) - beta*x
        solver = GradientDescent(fun=in_obj, acceleration=False, stepsize=1e-1, maxiter=2500, tol=1e-2)
        b_i, _ = solver.run(0.)

        # The final point is found where the line starting from the initial point,
        # whith slope beta, intersects the travel time function.
        # This point is found via a bisection
        
        fin_obj = lambda x: travel_time(x) - beta*(x - b_i) - travel_time(b_i)

        # Two points where to start the bisection are computed
        step = .5
        high = jax.lax.while_loop(lambda a: fin_obj(a) > 0, lambda a: a + step, b_i + step)
        low = high - step
        b_e = Bisection(fin_obj, low, high, tol=1e-2, check_bracket=False, jit=True).run().params

        # The interval extremes are returned
        return (b_i, b_e)
    
    def find_b0(t_a, travel_time):
        """ Given an arrival time, finds the maximal beta such that
        the arrival time is a kink equilibrium for every value higher than the one returned.

        Finds the parameter by bisection.
        """

        # A really low and a really high value are defined as starting points for the bisection
        min = 1e-1
        max = 1-min

        # The objective function, an indicator function that shows wether the parameter t_a
        # is in the interval for a given beta, is defined
        isin = lambda x, int: jnp.where(jnp.logical_and((x > int[0]), (x < int[1])), 1, -1)
        isin_obj = lambda b: isin(t_a, find_bs(b, travel_time))

        # If t_a is not in the interval for the starting points,
        # the starting points are returned themselves.
        # Otherwise, the bisection algorithm is run.
        is_max = isin_obj(min) == -1
        is_min = isin_obj(max) == 1
        sol = jnp.where(
            jnp.logical_and(
                jnp.logical_not(is_max),
                jnp.logical_not(is_min)),
            Bisection(isin_obj, min, max, tol=1e-3, check_bracket=False, jit=True).run().params,
            jnp.where(is_max, min, max))
        return sol
    
    b0 = find_b0(t_a, travel_time)
    return 1 - norm.cdf(b0, mu_b, sigma)

def likelihood_internal(travel_time, t_a, mu_b, sigma):
    travel_time_diff = jax.grad(travel_time)
    return norm.pdf(travel_time_diff(t_a), mu_b, sigma)
    
#%%

beta = .7
gamma = 1.3
n = 200
betas = jnp.array([beta]*n)
gammas = jnp.array([gamma]*n)
ts = jnp.linspace(7, 12, n)
plt.plot(ts, find_td(asymm_gaussian_plateau())(betas, gammas, ts), '.')
# plt.vlines([b_i, b_e], 7, 9)
plt.show()

#%%
num=200
betas, gammas, ts, t_as = generate_arrival(num, travel_time=asymm_gaussian_plateau())
#%%
liks = np.array([likelihood_kink(asymm_gaussian_plateau(), t_a, .7, .1) for t_a in tqdm(t_as)])
#%%

lik_int_restr = lambda t: likelihood_internal(asymm_gaussian_plateau(), t, .7, .1)
liks_int = vmap(lik_int_restr)(t_as)

#%%
liks_restr = lambda t: likelihood_kink(asymm_gaussian_plateau(), t, .7, .1)
start_time = time.time()
liks_new = vmap(liks_restr)(t_as)
total_time = time.time() - start_time
print(f"{total_time/num:.3} s per iteration, {total_time:.3} s in total")
#%%

fig, axs = plt.subplots(1, 2)
colors = cm.plasma(liks_int)
axs[0].scatter(np.random.normal(size=num), t_as, s=10, color=colors)
axs[1].scatter(ts, t_as, s=10, color=colors)
fig.show()
