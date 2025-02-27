from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.optimize import minimize, bisect
from scipy.stats import norm

import numpy as np

import jax
from jax import vmap
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

beta = .7
travel_time = asymm_gaussian_plateau()
def likelihood_kink(travel_time, t_a, mu_b, sigma):
    def find_bs(beta, travel_time):
        in_obj = lambda x: travel_time(x) - beta*x
        solver = GradientDescent(fun=in_obj, acceleration=False, stepsize=1e-1, maxiter=2500, tol=1e-2)

        b_i, _ = solver.run(0.)

        fin_obj = lambda x: travel_time(x) - beta*(x - b_i) - travel_time(b_i)
        step = .5
        high = jax.lax.while_loop(lambda a: fin_obj(a) > 0, lambda a: a + step, low + step)
        low = high - step
        b_e = Bisection(fin_obj, low, high, tol=1e-2, check_bracket=False, jit=True).run().params
        return (b_i, b_e)
    
    def find_b0(t_a, travel_time):
        min = 3e-1
        max = 1-min
        isin = lambda x, int: jnp.where(jnp.logical_and((x > int[0]), (x < int[1])), 1, -1)
        isin_obj = lambda b: isin(t_a, find_bs(b, travel_time))
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
    return 1 - jax.scipy.stats.norm.cdf(b0, mu_b, sigma)
    
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
liks_restr = lambda t: likelihood_kink(asymm_gaussian_plateau(), t, .7, .1)
start_time = time.time()
liks_new = vmap(liks_restr)(t_as)
total_time = time.time() - start_time
print(f"{total_time/num} per iteration, {total_time} in total")
#%%

fig, axs = plt.subplots(1, 2)
colors = cm.plasma(liks_new)
axs[0].scatter(np.random.normal(size=num), t_as, s=10, color=colors)
axs[1].scatter(ts, t_as, s=10, color=colors)
fig.show()
