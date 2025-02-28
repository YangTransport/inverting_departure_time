from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.optimize import minimize, bisect

import numpy as np

import jax
from jax import jit
from jax import vmap
from jax.scipy.integrate import trapezoid
from jax.scipy.stats import norm
import jax.numpy as jnp
from jaxopt import GradientDescent
from jaxopt import BFGS
from jaxopt import Bisection

from timeit import timeit
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

def likelihood(travel_time, t_a, mu_b, mu_t, sigma, sigma_t):
    """ Finds the likelihood of a point realizing a minimum,
    for the travel time, beta, gamma and t* distributions determined by the parameters.
    Beta, gamma and t* are assumed to be normally distributed.
    Gamma still has to be implemented
    """
    def find_bs(beta):
        """ Given a travel time function and a beta value,
        finds the interval in which the optimal arrival time is constant
        (and there are thus no kink minima).

        Returns a couple containing initial and final points of the interval
        """
        
        # A gradient descent algorithm finds the initial point
        in_obj = lambda x: travel_time(x) - beta*x
        solver = GradientDescent(fun=in_obj, acceleration=False, stepsize=1e-1, maxiter=2500)
        b_i, _ = solver.run(0.)

        # The final point is found where the line starting from the initial point,
        # whith slope beta, intersects the travel time function.
        # This point is found via a bisection
        
        fin_obj = lambda x: travel_time(x) - beta*(x - b_i) - travel_time(b_i)

        # Two points where to start the bisection are computed
        step = .5
        high = jax.lax.while_loop(lambda a: fin_obj(a) > 0, lambda a: a + step, b_i + step)
        low = high - step
        b_e = Bisection(fin_obj, low, high, check_bracket=False, jit=True).run().params

        # The interval extremes are returned
        return (b_i, b_e)
    
    def find_b0(t_a):
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
        isin_obj = lambda b: isin(t_a, find_bs(b))

        # If t_a is not in the interval for the starting points,
        # the starting points are returned themselves.
        # Otherwise, the bisection algorithm is run.
        is_max = isin_obj(min) == -1
        is_min = isin_obj(max) == 1
        sol = jnp.where(
            jnp.logical_and(
                jnp.logical_not(is_max),
                jnp.logical_not(is_min)),
            Bisection(isin_obj, min, max, check_bracket=False, jit=True).run().params,
            jnp.where(is_max, min, max))
        return sol

    # For computing the probability that a point is a kink minimum,
    # an integral is computed as in the latex.
    b0 = find_b0(t_a)
    likelihood_kink = norm.pdf(t_a, mu_t, sigma_t) * (1 - norm.cdf(b0, mu_b, sigma))

    # Now for the internal minimum:
    # the easiest probability to compute is the probability that
    # a point is allowed to be an internal minimum for some realization of beta
    travel_time_diff = jax.grad(travel_time)
    prob_allowed = norm.pdf(travel_time_diff(t_a), mu_b, sigma) * (jax.grad(travel_time_diff)(t_a) > 0)

    def inner_int(b):
        bs = find_bs(b)
        return (norm.cdf(bs[1], mu_t, sigma_t) - norm.cdf(bs[0], mu_t, sigma_t)) * norm.pdf(b, mu_b, sigma)
    
    min = 1e-2
    x = jnp.linspace(min, 1-min, 50)
    fx = vmap(inner_int)(x)
    int_result = trapezoid(fx, x)

    likelihood_internal = int_result * prob_allowed

    return likelihood_kink + likelihood_internal

def total_liks(travel_time, t_as):
    def mapped_lik(mu_b, mu_t, sigma, sigma_t):
        lik_restr = lambda t_a: likelihood(travel_time, t_a, mu_b, mu_t, sigma, sigma_t)
        return vmap(lik_restr)(t_as)
    return mapped_lik

def total_log_lik(travel_time, t_as):
    def mapped_lik(mu_b, mu_t, sigma, sigma_t):
        lik_restr = lambda t_a: likelihood(travel_time, t_a, mu_b, mu_t, sigma, sigma_t)
        return jnp.sum(jnp.log(vmap(lik_restr)(t_as)))
    return mapped_lik
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
num=1000
betas, gammas, ts, t_as = generate_arrival(num, travel_time=asymm_gaussian_plateau(), mu_gamma=1000)
lik_fun = jit(lambda mus: -total_log_lik(asymm_gaussian_plateau(), t_as)(mus[0], mus[1], mus[2], mus[3]))

#%%

res = minimize(lik_fun, (10, 10, .5, .5), method="Nelder-Mead")
print(res.x)
#%%

solver = GradientDescent(lik_fun, verbose=True)
init = (10., 10., .5, .5)
val, state = solver.run(init)
print("finished optimizing")
print(val)
#%%

tot_liks = vmap(lambda t: likelihood_kink(asymm_gaussian_plateau(), t, .7, 9.5, .1, 1.) + likelihood_internal(asymm_gaussian_plateau(), t, .7, 9.5, .1, 1.))
liks = tot_liks(t_as)
#%%

liks = total_liks(asymm_gaussian_plateau(), t_as)(.7, 9.5, .1, 1.)

#%%
fig, axs = plt.subplots(1, 2)
colors = cm.plasma(liks/max(liks))
axs[0].scatter(np.random.normal(size=num), t_as, s=10, color=colors)
axs[1].scatter(ts, t_as, s=10, color=colors)
fig.show()
