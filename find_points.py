from jaxopt import GradientDescent, Bisection
import jax.numpy as jnp
from jax.lax import while_loop
from utils import steps

def find_bs(beta, travel_time):
    """ Given a travel time function and a beta value,
    finds the interval in which the optimal arrival time is constant
    (and there are thus no kink minima).

    Returns a couple containing initial and final points of the interval
    """
    
    # A gradient descent algorithm finds the initial point
    stepsize = steps()
    in_obj = lambda x: travel_time(x) - beta*x
    solver = GradientDescent(fun=in_obj, acceleration=False, stepsize=stepsize)
    b_i, _ = solver.run(0.)

    # The final point is found where the line starting from the initial point,
    # whith slope beta, crosses the travel time function.
    # This point is found via a bisection
        
    fin_obj = lambda x: travel_time(x) - beta*(x - b_i) - travel_time(b_i)

    # Two points where to start the bisection are computed
    step = .5
    high = while_loop(lambda a: fin_obj(a) > 0, lambda a: a + step, b_i + step)
    low = high - step
    b_e = Bisection(fin_obj, low, high, check_bracket=False).run().params

    # The interval extremes are returned
    return (b_i, b_e)
    
def find_b0(t_a, travel_time):
    """ Given an arrival time, finds the maximal beta such that
    the arrival time is a kink equilibrium for every value higher than the one returned.
    
    Finds the parameter by bisection.
    """

    # A really low and a really high value are defined as starting points for the bisection
    min = 1e-1
    max = 1

    # The objective function, an indicator function that shows wether the parameter t_a
    # is in the interval for a given beta, is defined
    isin = lambda x, int: jnp.where(jnp.logical_and((x > int[0]), (x < int[1])), 1, -1)
    isin_obj = lambda b: isin(t_a, find_bs(b, travel_time))

    # If t_a is not in the interval for the starting points,
    # the starting points are returned themselves.
    # Otherwise, the bisection algorithm is run.
    is_max = isin_obj(min) == -1
    is_min = isin_obj(max) == 1
    sol = jnp.where(
        jnp.logical_and(
            jnp.logical_not(is_max),
            jnp.logical_not(is_min)),
        Bisection(isin_obj, min, max, check_bracket=False).run().params,
        jnp.where(is_max, min, max))
    return sol


def find_gs(gamma, travel_time):
    """ Given a travel time function and a gamma value,
    finds the interval in which the optimal arrival time is constant
    (and there are thus no kink minima).

    Returns a couple containing initial and final points of the interval
    """
    
    # A gradient descent algorithm finds the final point
    stepsize = steps()
    fin_obj = lambda x: travel_time(x) + gamma*x
    solver = GradientDescent(fun=fin_obj, acceleration=False, stepsize=stepsize, maxiter=2500)
    g_e, _ = solver.run(24.)

    # The initial point is found where the line starting from the
    # final point, whith slope -gamma, crosses the travel time
    # function.
    # This point is found via a bisection
        
    fin_obj = lambda x: travel_time(x) + gamma*(x - g_e) - travel_time(g_e)

    # Two points where to start the bisection are computed
    step = .5
    low = while_loop(lambda a: fin_obj(a) > 0, lambda a: a - step, g_e - step)
    high = low + step
    g_i = Bisection(fin_obj, low, high, check_bracket=False).run().params

    # The interval extremes are returned
    return (g_i, g_e)

def find_g0(t_a, travel_time):
    """Given an arrival time, finds the maximal gamma such that the
    arrival time is a kink equilibrium for every value lower than the
    one returned.
    
    Finds the parameter by bisection.

    """

    # A really low and a really high value are defined as starting
    # points for the bisection
    min = 1
    max = 20

    # The objective function, an indicator function that shows wether
    # the parameter t_a is in the interval for a given gamma, is
    # defined
    isin = lambda x, int: jnp.where(jnp.logical_and((x > int[0]), (x < int[1])), 1, -1)
    isin_obj = lambda g: isin(t_a, find_gs(g, travel_time))

    # If t_a is not in the interval for the starting points,
    # the starting points are returned themselves.
    # Otherwise, the bisection algorithm is run.
    is_max = isin_obj(min) == -1
    is_min = isin_obj(max) == 1
    sol = jnp.where(
        jnp.logical_and(
            jnp.logical_not(is_max),
            jnp.logical_not(is_min)),
        Bisection(isin_obj, min, max, check_bracket=False).run().params,
        jnp.where(is_max, min, max))
    return sol
