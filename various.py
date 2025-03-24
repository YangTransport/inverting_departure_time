from matplotlib import pyplot as plt

from scipy.optimize import minimize, bisect

import jax
jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)
import jax.numpy as jnp
from jax import jit, vmap
from jaxopt import GradientDescent, ProjectedGradient
from jaxopt import Bisection
from jaxopt.projection import projection_box

from timeit import timeit
import time
from tqdm import tqdm

from generate_data import generate_arrival
from travel_times import asymm_gaussian_plateau
from retrieve_data import likelihood, total_log_lik, total_liks
from utils import TravelTime

import numpy as np
#%%
num = 1000
tt = TravelTime(asymm_gaussian_plateau())
par = (.4, 1.4, 10., .1, 1.5)
betas, gammas, ts, t_as = generate_arrival(num, tt, *par)

@jit
def lik_fun(par):
    log_lik = total_log_lik(tt, t_as)(*par)
    return -log_lik

def obj_fun(par):
    print(("{:8.4f}"*5).format(*par))
    return lik_fun(par)
#%%

g_betas = jnp.linspace(.01, .9, 6)
g_gammas = jnp.linspace(1, 4, 4)
g_ts = jnp.linspace(6, 11, 3)
g_sigmas = jnp.linspace(.1, .5, 5)
g_sigmats = jnp.linspace(1, 4, 3)

mesh_par = jnp.meshgrid(g_betas, g_gammas, g_ts, g_sigmas, g_sigmats)

# vec_lik = jit(vmap(vmap(vmap(vmap(vmap(total_log_lik(tt, t_as), (None, None, None, None, 0)), (None, None, None, 0, None)), (None, None, 0, None, None)), (None, 0, None, None, None)), (0, None, None, None, None)))

# NOTE I put here (1, 1, 1, 1, 1) but this should definitely looked at
# better. It looks like it works right now, but I don't really
# understand why

vec_lik = vmap(total_log_lik(tt, t_as), (1, 1, 1, 1, 1))

# grid_result = vec_lik(g_betas, g_gammas, g_ts, g_sigmas, g_sigmats)

grid_result = vec_lik(*mesh_par)

best = jnp.array(jnp.unravel_index(grid_result.argmax(), grid_result.shape))
init = jnp.array([g_betas[best[0]], g_gammas[best[1]], g_ts[best[2]], g_sigmas[best[3]], g_sigmats[best[4]]])

print("Initial values are")
print(init)

start_time = time.time()
res = minimize(obj_fun, init, method="Nelder-Mead")
print(res.x)
error = jnp.mean(np.abs(res.x - par)/par)
print(f"error {error}")
print(f"{time.time() - start_time} seconds")
#%%

start_time = time.time()
# solver = ProjectedGradient(lik_fun, projection_box, verbose=True)
# val, state = solver.run(init, hyperparams_proj=(jnp.array([1e-2, 1, 0, 0, 0]), jnp.array([tt.maxb, tt.maxg, 24, 1, 5])))

solver = GradientDescent(lik_fun, verbose=True)
val, state = solver.run(init)
print(jnp.array(val))
print("finished optimizing")
print(f"{time.time() - start_time} seconds")
#%%
x = jnp.linspace(6, 15, 500)
liks_x_real = total_liks(tt, x)(*par)
liks_x_found = total_liks(tt, x)(*res.x)
#%%
h = 100

plt.hist(t_as, 100)
plt.fill_between(x, liks_x_real*h, alpha=.3, color="red", label='Real likelihood')
plt.fill_between(x, liks_x_found*h, alpha=.3, color="green", label='Retrieved likelihood')
plt.xlim(6, 13)
# plt.ylim(0, 400)
plt.legend()
# plt.savefig("latex/img/likelihood_over_ta.png")
plt.show()
#%%
liks = total_liks(tt, t_as)(*par)

#%%
fig, axs = plt.subplots(1, 2)
# dis = axs[0].scatter(np.random.normal(size=num), t_as, s=10, c=liks/max(liks))
dis = axs[0].scatter(betas, t_as, s=10, c=ts/max(ts))
ord = axs[1].scatter(ts, t_as, s=10, c=liks/max(liks))
axs[0].set_ylabel("t_a")
axs[1].set_ylabel("t_a")
axs[1].set_xlabel("t*")
# axs[0].set_ylim(6, 7)
fig.colorbar(ord, shrink=.8, ticks=[])
fig.show()

#%%
ll_actual = lambda x, y: total_log_lik(tt, t_as)(x, y, *par[2:])
ll_optimized = lambda x, y: total_log_lik(tt, t_as)(x, y, *res.x[2:])
betas_contour =jnp.linspace(.01, .99, 200)
gammas_contour = jnp.linspace(1.01, 4, 200)
m_contour = jnp.meshgrid(betas_contour, gammas_contour)
matrix_actual = vmap(vmap(ll_actual, (0, None)), (None, 0))(betas_contour, gammas_contour) # vmap(ll_actual)(*m_contour)
matrix_optimized = vmap(vmap(ll_optimized, (0, None)), (None, 0))(betas_contour, gammas_contour) # vmap(ll_optimized)(*m_contour)
#%%
X, Y = jnp.meshgrid(betas_contour, gammas_contour)
plt.contour(betas_contour, gammas_contour, matrix_actual, levels=50)
plt.plot(par[0], par[1], 'or')
plt.plot(res.x[0], res.x[1], 'og')
plt.xlabel("mu_b")
plt.ylabel("mu_g")
# plt.savefig("contour_mus_not_converging.png")
plt.show()
