import jax.numpy as jnp
from jax import jit
from jax import vmap
from jaxopt import GradientDescent
from scipy.stats import truncnorm
from scipy.stats import norm

#%%

def travel_time(t_a, mu=9.5, sigma=1):
    return 1/sigma/jnp.sqrt(2*jnp.pi)*jnp.exp(-((t_a-mu)**2)/2*sigma**2)

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
    
    def cost(t_a, beta, gamma, t_star):
        return travel_time(t_a) + beta * jnp.maximum(0, t_star - t_a) + gamma * jnp.maximum(0, t_a - t_star)
    cost = jit(cost)

    # A gradient descent optimizer finds the left minimum (if there is one)

    def find_td(beta, gamma, t_star):
        solver = GradientDescent(fun=cost, acceleration=False)
        val, state = solver.run(0., beta, gamma, t_star)
        return jnp.where(cost(val, beta, gamma, t_star) < cost(t_star, beta, gamma, t_star), val, t_star)
    find_tds = vmap(find_td)
    
    return find_tds(betas, gammas, ts)
