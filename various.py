from matplotlib import pyplot as plt
from matplotlib import cm

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

#%%
num=1000
betas, gammas, ts, t_as = generate_arrival(num, travel_time=asymm_gaussian_plateau(), mu_gamma=1000)
lik_fun = jit(lambda mus: -total_log_lik(asymm_gaussian_plateau(), t_as)(mus[0], mus[1], mus[2], mus[3]))

#%%

res = minimize(lik_fun, (.2, 7, .5, .5), method="Nelder-Mead")
print(res.x)
#%%

solver = GradientDescent(lik_fun, stepsize=1e-5, maxiter=2000)
init = (1., 10., .5, .5)
val, state = solver.run(init)
print("finished optimizing")
print(jnp.array(val))
#%%
x = jnp.linspace(6, 13, 1000)
liks_x = total_liks(asymm_gaussian_plateau(), x)(.7, 9.5, .1, 1)
#%%
h = 120
plt.hist(t_as, 80)
plt.fill_between(x, liks_x*h, alpha=.3, color="red")
plt.show()
#%%

liks = total_liks(asymm_gaussian_plateau(), t_as)(.7, 9.5, .1, 1.)

#%%
fig, axs = plt.subplots(1, 2)
colors = cm.plasma(liks/max(liks))
axs[0].scatter(np.random.normal(size=num), t_as, s=10, color=colors)
axs[1].scatter(ts, t_as, s=10, color=colors)
fig.show()
