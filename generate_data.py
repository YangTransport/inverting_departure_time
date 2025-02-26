import jax.numpy as jnp
from jax import jit
from jax import vmap
import jax
from jaxopt import GradientDescent
from scipy.stats import truncnorm
from scipy.stats import norm

#%%

def travel_time(t_a, mu=9.5, sigma=.2):
    return 1/sigma/jnp.sqrt(2*jnp.pi)*jnp.exp(-((t_a-mu)**2)/2/sigma**2)

def cost(travel_time):
    def inner_cost(t_a, beta, gamma, t_star):
        return travel_time(t_a) + beta * jnp.maximum(0, t_star - t_a) + gamma * jnp.maximum(0, t_a - t_star) 
    return jit(inner_cost)

# A gradient descent optimizer finds the left minimum (if there is one)                                  

def find_td(travel_time):
    def inner_find_td(beta, gamma, t_star):
        cost_fun = cost(travel_time)
        solver = GradientDescent(fun=cost_fun, acceleration=False)
        lval, _ = solver.run(0., beta, gamma, t_star)
        rval, _ = solver.run(24., beta, gamma, t_star)
        val = jnp.where(cost_fun(rval, beta, gamma, t_star) < cost_fun(lval, beta, gamma, t_star), rval, lval)
        return jnp.where(cost_fun(val, beta, gamma, t_star) < cost_fun(t_star, beta, gamma, t_star), val, t_star)
    return vmap(inner_find_td)

def generate_arrival(n, mu_beta=0.7, mu_gamma=1.3, mu_t=9.5, sigma=0.1, sigma_t=1, travel_time=travel_time):
    """Generate samples of departure time.

    Arguments:
        n: number of samples
        mu_beta, mu_gamma, mu_t: mean for the parameters beta, gamma and t*
        sigma: variance for the parameters beta and gamma
        sigma_t: variance for the parameter t*

    Returns a numpy array with the data
    """

    # Betas, gammas and t_star are generated according to the chosen distributions
    
    betas = truncnorm.rvs(-mu_beta / sigma, 100000, loc=mu_beta, scale=sigma, size=n)
    gammas = truncnorm.rvs(-mu_gamma / sigma, 100000, loc=mu_gamma, scale=sigma, size=n)
    ts = norm.rvs(mu_t, sigma_t, n)
    return betas, gammas, ts, find_td(travel_time)(betas, gammas, ts)

