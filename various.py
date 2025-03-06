from matplotlib import pyplot as plt

from scipy.optimize import minimize, bisect


import jax.numpy as jnp
from jax import jit
from jaxopt import GradientDescent
from jaxopt import Bisection

from timeit import timeit
import time
from tqdm import tqdm

from generate_data import generate_arrival
from travel_times import asymm_gaussian_plateau
from retrieve_data import likelihood, total_log_lik, total_liks

import numpy as np
#%%
num=1000
par = (.3, 2.4, 10, .2, 1)
betas, gammas, ts, t_as = generate_arrival(num, asymm_gaussian_plateau(), *par)

@jit
def lik_fun(par):
    log_lik = total_log_lik(asymm_gaussian_plateau(), t_as)(*par)
    return -log_lik
#%%

def obj_fun(par):
    print(("{:8.4f}"*5).format(*par))
    return lik_fun(par)
#%%
g_betas = jnp.linspace(.01, .9, 5)
g_gammas = jnp.linspace(1, 4, 5)
g_ts = jnp.linspace(6, 11, 3)
g_sigmas = jnp.linspace(.1, .5, 5)
g_sigmats = jnp.linspace(1, 4, 3)

vec_lik = jit(vmap(vmap(vmap(vmap(vmap(total_log_lik(asymm_gaussian_plateau(), t_as), (None, None, None, None, 0)), (None, None, None, 0, None)), (None, None, 0, None, None)), (None, 0, None, None, None)), (0, None, None, None, None)))

grid_result = vec_lik(g_betas, g_gammas, g_ts, g_sigmas, g_sigmats)
best = jnp.array(jnp.unravel_index(grid_result.argmax(), grid_result.shape))
init = jnp.array([g_betas[best[0]], g_gammas[best[1]], g_ts[best[2]], g_sigmas[best[3]], g_sigmats[best[4]]])

print("Initial values are")
print(init)

start_time = time.time()
res = minimize(lik_fun, (.9, 2., 10., .5, .5), method="Nelder-Mead")
print(res.x)
print(f"{time.time() - start_time} seconds")
#%%
start_time = time.time()
solver = GradientDescent(lik_fun, verbose=True, maxiter=150, maxls=30)
init = (.9, 2., 10., .5, .5)
val, state = jit(solver.run)(init)
print(jnp.array(val))
print("finished optimizing")
print(f"{time.time() - start_time} seconds")
#%%
x = jnp.linspace(6, 12, 500)
liks_x = total_liks(asymm_gaussian_plateau(), x)(.7, 1.3, 9.5, .1, 1.)
#%%
h = 120
plt.hist(t_as, 80)
plt.fill_between(x, liks_x*h, alpha=.3, color="red")
plt.show()
#%%
liks = total_liks(asymm_gaussian_plateau(), t_as)(*par)

#%%
fig, axs = plt.subplots(1, 2)
dis = axs[0].scatter(np.random.normal(size=num), t_as, s=10, c=liks/max(liks))
ord = axs[1].scatter(ts, t_as, s=10, c=liks/max(liks))
axs[0].set_ylabel("t_a")
axs[1].set_ylabel("t_a")
axs[1].set_xlabel("t*")
fig.colorbar(ord, shrink=.8, ticks=[])
fig.show()
