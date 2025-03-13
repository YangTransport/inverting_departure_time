from jax import grad, vmap
from jax.nn import relu
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
    # cdf_b = lambda b: jnorm.cdf(b, mu_b, sigma)
    # cdf_g = lambda g: jnorm.cdf(g, mu_g, sigma)
    # pdf_b = lambda b: jnorm.pdf(b, mu_b, sigma)
    # pdf_g = lambda g: jnorm.pdf(g, mu_g, sigma)
    cdf_b = lambda b: jtruncnorm.cdf(b, -mu_b / sigma, (1 - mu_b) / sigma, loc=mu_b, scale=sigma)
    cdf_g = lambda g: jtruncnorm.cdf(g, (1 -mu_g) / sigma, 100000, loc=mu_g, scale=sigma)
    pdf_b = lambda b: jtruncnorm.pdf(b, -mu_b / sigma, (1 - mu_b) / sigma, mu_b, sigma)
    pdf_g = lambda g: jtruncnorm.pdf(g, (1 - mu_g) / sigma, 10000, mu_g, sigma)

    # For computing the probability that a point is a kink minimum, an
    # integral is computed as in the latex.
    b0 = find_b0(t_a, travel_time)
    g0 = find_g0(t_a, travel_time)
    likelihood_kink = jnorm.pdf(t_a, mu_t, sigma_t) * (1 - cdf_b(b0)) * (1 - cdf_g(g0))

    # Now for the internal minima: the easiest probability to compute
    # is the probability that a point is allowed to be an internal
    # minimum for some realization of beta

    lower_inner_int_b = lambda t: (lambda s: jnorm.pdf(s, mu_t, sigma_t)
                                 * (t < find_b0(s, travel_time)))

    t_points = 800
    ts = jnp.linspace(0, 24, t_points)

    prob_lower_b = lambda t: trapezoid(vmap(lower_inner_int_b(t))(ts), ts, axis=0)
    normalization_term_b = trapezoid(vmap(pdf_b)(ts) * vmap(prob_lower_b)(ts), ts, axis=0)
    conditional_pdf_b = pdf_b(travel_time.df(t_a)) * prob_lower_b(travel_time.df(t_a)) / normalization_term_b
    prob_allowed_b = conditional_pdf_b * relu(travel_time.d2f(t_a))

    lower_inner_int_g = lambda t: (lambda s: jnorm.pdf(s, mu_t, sigma_t)
                                   * (t < find_g0(s, travel_time)))

    prob_lower_g = lambda t: trapezoid(vmap(lower_inner_int_g(t))(ts), ts, axis=0)
    normalization_term_g = trapezoid(vmap(pdf_g)(ts) * vmap(prob_lower_g)(ts), ts, axis=0)
    conditional_pdf_g = pdf_g(-travel_time.df(t_a)) * prob_lower_g(-travel_time.df(t_a)) / normalization_term_g
    prob_allowed_g = conditional_pdf_g * relu(travel_time.d2f(t_a))


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
    points = 200
    x_b = jnp.linspace(min_b, travel_time.maxb, points)
    fx_b = vmap(inner_int_b)(x_b)
    int_result_b = trapezoid(fx_b, x_b, axis=0)

    min_g = 1
    
    x_g = jnp.linspace(min_g, travel_time.maxg, points)
    fx_g = vmap(inner_int_g)(x_g)
    int_result_g = trapezoid(fx_g, x_g, axis=0)
    likelihood_internal = int_result_b * prob_allowed_b + int_result_g * prob_allowed_g
    likelihood = likelihood_kink + likelihood_internal
    return jnp.maximum(likelihood, 1e-31)

def total_liks(travel_time, t_as):
    def mapped_lik(mu_b, mu_g, mu_t, sigma, sigma_t):
        lik_restr = lambda t_a: likelihood(travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t)
        return vmap(lik_restr)(t_as)
    return mapped_lik

def total_log_lik(travel_time, t_as):
    def mapped_lik(mu_b, mu_g, mu_t, sigma, sigma_t):
        lik_restr = lambda t_a: likelihood(travel_time, t_a, mu_b, mu_g, mu_t, sigma, sigma_t)
        return jnp.sum(jnp.log(vmap(lik_restr)(t_as)), axis=0)
    return mapped_lik
