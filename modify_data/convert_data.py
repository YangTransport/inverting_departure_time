import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
from instant_to_tt import arrival_time
from load_data import load_data

x, travel_times = load_data()
#%%
dps = [2, 3, 4]
new_tt = dict()
for dp in dps:
    new_tt[dp] = np.array([arrival_time(d, dp, x, 1/travel_times) for d in x]) - x

plt.plot(x, travel_times, label="Original data")
for dp in dps:
    plt.plot(x, new_tt[dp], label=r"Transformed data, $d^p = {}$".format(dp), linewidth=.8)
plt.plot([0, 7], [0, 7*.61], label=r"Tipical slope $y = \frac{\beta}{\alpha} x = 0.61 x$, for reference")
plt.plot([24, 21], [0, 2*2.35], label=r"Tipical slope $y = \frac{\gamma}{\alpha} x = -2.38 x$, for reference")
plt.legend()
plt.xlabel("Time of the day (hours)")
plt.ylabel("Travel time (hours)")
plt.show()

#%%
d_travel_times = (travel_times[:-1] - travel_times[1:]) / (x[:-1] - x[1:])
d_new_tt = (new_tt[:-1] - new_tt[1:]) / (x[:-1] - x[1:])
plt.plot(x[1:], d_travel_times)
plt.plot(x[1:], d_new_tt)
plt.hlines([0.6, -1.4], x[0], x[-1], 'r')
plt.show()

