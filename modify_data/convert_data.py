import matplotlib.pyplot as plt
import numpy as np
from instant_to_tt import arrival_time
from load_data import load_data

x, travel_times = load_data()

#%%
d_travel_times = (travel_times[:-1] - travel_times[1:]) / (x[1] - x[0])
plt.plot(x[1:], d_travel_times)
plt.hlines([0.6, -1.4], x[0], x[-1], 'r')
plt.show()

#%%
new_tt = np.array([arrival_time(d, 2, x, 1/travel_times) for d in x]) - x

plt.plot(x, new_tt)
plt.plot(x, travel_times)
plt.show()
