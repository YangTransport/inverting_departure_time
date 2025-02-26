import generate_data
from matplotlib import pyplot as plt
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
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
gamma = 1.3
n = 200
betas = jnp.array([beta]*n)
gammas = jnp.array([gamma]*n)
ts = jnp.linspace(7, 12, n)
plt.plot(ts, find_td(asymm_gaussian_plateau(sigma_l=.5, plateau_len=1))(betas, gammas, ts), '.')
plt.show()

#%%

def log_lik_ols(beta, gamma, t_star, t_a):
    n = len(t_a)
    best = find_td(asymm_gaussian())(jnp.array([beta]), jnp.array([gamma]), jnp.array([t_star]))
    return -jnp.sum((best - t_a)**2)

n=200
_, _, _, t_a = generate_arrival(n)

obj = lambda b: -log_lik_ols(b[0], b[1], b[2], t_a)

min = minimize(obj, (1, 1, 9), method='nelder-mead')

#%%
beta, gamma, t_star = min.x
plt.plot(np.random.normal(size=n), t_a, '.b')
plt.plot(0, find_td(asymm_gaussian())(jnp.array([beta]), jnp.array([gamma]), jnp.array([t_star])), 'or')
plt.plot(0, np.mean(t_a), '.g')
plt.show()

#%%
num=200
betas, gammas, ts, t_as = generate_arrival(num, travel_time=asymm_gaussian())
fig, axs = plt.subplots(1, 2)
axs[0].plot(np.random.normal(size=num), t_as, '.')
axs[1].plot(ts, t_as, '.')
fig.show()
