import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from generate_data import  generate_arrival
from travel_times import asymm_gaussian_plateau
from utils import TravelTime
from find_points import find_bs, find_gs

tt = TravelTime(asymm_gaussian_plateau())

num = 1000
par = (.7, 1.2, 9.5, .1, 1)
np.random.seed(14)
betas, gammas, ts, t_as = generate_arrival(num, tt, *par)

bs = find_bs(par[0], tt)
gs = find_gs(par[1], tt)

#%%
x = np.random.normal(size=num)
fig_scatter, ax_scatter = plt.subplots(figsize=(4, 5))
b_low = bs[0] - par[3]*1.7
b_high =bs[0] + par[3]*1.7
g_low = gs[1] - par[3]/2
g_high = gs[1] + par[3]/2
ax_scatter.fill_between([x.min(), x.max()], [b_low]*2, [b_high]*2, alpha=.1, color="red")
ax_scatter.fill_between([x.min(), x.max()], [g_low]*2, [g_high]*2, alpha=.1, color="green")
ax_scatter.scatter(x, t_as, s=2)
ax_scatter.set_xlim(x.min(), x.max())
ax_scatter.set_ylim(4, 15)
fig_scatter.savefig("slides/img/t_as.png", dpi=600)

y = np.linspace(4, 15, 500)
ax_scatter.plot(tt.f(y)*4 + x.min(), y, color="red")
fig_scatter.savefig("slides/img/t_as_tt.png", dpi=600)

#%%

fig_bin, ax_bin = plt.subplots(figsize=(6, 4))
n, bins, patches = ax_bin.hist(t_as, 80)
# ax_bin.fill_betweenx([ax_bin.get_ylim()[0], ax_bin.get_ylim()[1]], [b_high]*2, [b_low]*2, color='red', alpha=.2)
# ax_bin.fill_betweenx([ax_bin.get_ylim()[0], ax_bin.get_ylim()[1]], [g_high]*2, [g_low]*2, color='green', alpha=.2)
for b, p in zip(bins, patches):
    if b < b_high and b > b_low - (bins[1] - bins[0]):
        p.set_facecolor("red")
    if b < g_high and b > g_low - (bins[1] - bins[0]):
        p.set_facecolor("green")

x = np.linspace(6, 13, 300)
tt_line = ax_bin.plot(x, tt.f(x)*40, color="purple", linewidth=2, label="Travel time function")

labels = [tt_line[0].get_label(), "Early arrivals", "Late arrivals"]
handles = [tt_line[0], Patch(facecolor="red"), Patch(facecolor="green")]
ax_bin.legend(handles, labels)
fig_bin.savefig("slides/img/t_as_bins_tt.png", dpi=600)
