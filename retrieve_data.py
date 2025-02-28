from jax import grad, vmap
from jax.scipy.integrate import trapezoid
from jax.scipy.stats import truncnorm, norm
import jax.numpy as jnp

from find_points import find_bs, find_b0

def likelihood(travel_time, t_a, mu_b, mu_t, sigma, sigma_t):
    """Finds the likelihood of a point realizing a minimum, for the
    travel time, beta, gamma and t* distributions determined by the
    parameters.  Beta, gamma and t* are assumed to be normally
    distributed.  Gamma still has to be implemented

    """

    # For computing the probability that a point is a kink minimum, an
    # integral is computed as in the latex.
    b0 = find_b0(t_a, travel_time)
    likelihood_kink = norm.pdf(t_a, mu_t, sigma_t) * (1 - norm.cdf(b0, mu_b, sigma))

    # Now for the internal minima: the easiest probability to compute
    # is the probability that a point is allowed to be an internal
    # minimum for some realization of beta
    travel_time_diff = grad(travel_time)
    prob_allowed = norm.pdf(travel_time_diff(t_a), mu_b, sigma) * (grad(travel_time_diff)(t_a) > 0)

    # This probability has to be mutiplied to the probability that t*
    # is actually in the interval that would yield a constant minimum,
    # that is itself the probability of beta yielding a certain
    # interval times the probability of t* being in that interval.
    def inner_int(b):
        bs = find_bs(b, travel_time)
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
