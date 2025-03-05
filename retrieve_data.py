from jax import grad, vmap
from jax.scipy.integrate import trapezoid
from jax.scipy.stats import truncnorm as jtruncnorm
from jax.scipy.stats import norm as jnorm
import jax.numpy as jnp

from find_points import find_bs, find_gs, find_b0, find_g0

def likelihood(travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t):
    """Finds the likelihood of a point realizing a minimum, for the
    travel time, beta, gamma and t* distributions determined by the
    parameters.  Beta, gamma and t* are assumed to be normally
    distributed.

    """

    # The truncated normals pdf and cdf are definde here
    cdf_b = lambda b: jnorm.cdf(b, mu_b, sigma)
    cdf_g = lambda g: jnorm.cdf(g, mu_g, sigma)
    pdf_b = lambda b: jnorm.pdf(b, mu_b, sigma)
    pdf_g = lambda g: jnorm.pdf(g, mu_g, sigma)
    # cdf_b = lambda b: truncjnorm.cdf(b, -mu_b / sigma, (1 - mu_b) / sigma, loc=mu_b, scale=sigma)
    # cdf_g = lambda g: truncjnorm.cdf(g, (1 -mu_g) / sigma, 100000, loc=mu_g, scale=sigma)
    # pdf_b = lambda b: truncjnorm.pdf(b, -mu_b / sigma, (1 - mu_b) / sigma, mu_b, sigma)
    # pdf_g = lambda g: truncjnorm.pdf(g, (1 - mu_g) / sigma, 10000, mu_g, sigma)

    # For computing the probability that a point is a kink minimum, an
    # integral is computed as in the latex.
    b0 = find_b0(t_a, travel_time)
    g0 = find_g0(t_a, travel_time)
    likelihood_kink = jnorm.pdf(t_a, mu_t, sigma_t) * (1 - cdf_b(b0)) * (1 - cdf_g(g0))

    # Now for the internal minima: the easiest probability to compute
    # is the probability that a point is allowed to be an internal
    # minimum for some realization of beta
    travel_time_diff = grad(travel_time)
    prob_allowed_b = pdf_b(travel_time_diff(t_a)) * (grad(travel_time_diff)(t_a) > 0)
    prob_allowed_g = pdf_g(-travel_time_diff(t_a)) * (grad(travel_time_diff)(t_a) > 0)

    # This probability has to be mutiplied to the probability that t*
    # is actually in the interval that would yield a constant minimum,
    # that is itself the probability of beta yielding a certain
    # interval times the probability of t* being in that interval.
    def inner_int_b(b):
        bs = find_bs(b, travel_time)
        return (jnorm.cdf(bs[1], mu_t, sigma_t) - jnorm.cdf(bs[0], mu_t, sigma_t)) * pdf_b(b)
    
    def inner_int_g(g):
        gs = find_gs(g, travel_time)
        return (jnorm.cdf(gs[1], mu_t, sigma_t) - jnorm.cdf(gs[0], mu_t, sigma_t)) * pdf_g(g)
    
    min_b = 1e-2
    points = 50
    x_b = jnp.linspace(min_b, 1-min_b, points)
    fx_b = vmap(inner_int_b)(x_b)
    int_result_b = trapezoid(fx_b, x_b)

    min_g = 1
    max_g = 15
    x_g = jnp.linspace(min_g, max_g, points)
    fx_g = vmap(inner_int_g)(x_g)
    int_result_g = trapezoid(fx_g, x_g)
    likelihood_internal = int_result_b * prob_allowed_b + int_result_g * prob_allowed_g
    return likelihood_kink + likelihood_internal

def total_liks(travel_time, t_as):
    def mapped_lik(mu_b, mu_g, mu_t, sigma, sigma_t):
        lik_restr = lambda t_a: likelihood(travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t)
        return vmap(lik_restr)(t_as)
    return mapped_lik

def total_log_lik(travel_time, t_as):
    def mapped_lik(mu_b, mu_g, mu_t, sigma, sigma_t):
        lik_restr = lambda t_a: likelihood(travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t)
        return jnp.sum(jnp.log(vmap(lik_restr)(t_as)))
    return mapped_lik
