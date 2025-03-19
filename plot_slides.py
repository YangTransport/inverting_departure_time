import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Arc
import numpy as np
from generate_data import  generate_arrival
from travel_times import asymm_gaussian_plateau
from utils import TravelTime
from find_points import find_bs, find_gs
from retrieve_data import likelihood, total_log_lik
from jax import vmap
import jax.numpy as jnp

#%%

tt = TravelTime(asymm_gaussian_plateau(sigma_l=.7, sigma_r=.4, plateau_len=.4))

num = 1000
par = [.8, 1.4, 9.5, .3, 1]
np.random.seed(144)
betas, gammas, ts, t_as = generate_arrival(num, tt, *par)

bs = find_bs(par[0], tt)
gs = find_gs(par[1], tt)

early_color = "green"
late_color="red"
tt_color = "purple"
#%%
x = np.random.normal(size=num)
fig_scatter, ax_scatter = plt.subplots(figsize=(4, 5))
b_low = bs[0] - par[3]*1.7
b_high =bs[0] + par[3]*1.7
g_low = gs[1] - par[3]/2
g_high = gs[1] + par[3]/2
ax_scatter.fill_between([x.min(), x.max()], [b_low]*2, [b_high]*2, alpha=.1, color=early_color)
ax_scatter.fill_between([x.min(), x.max()], [g_low]*2, [g_high]*2, alpha=.1, color=late_color)
ax_scatter.scatter(x, t_as, s=2)
ax_scatter.set_xlim(x.min(), x.max())
ax_scatter.set_ylim(4, 15)
ax_scatter.set_xticks([])
ax_scatter.set_ylabel(r"$t_a$ (h)")
fig_scatter.savefig("slides/img/t_as.png", dpi=600)

y = np.linspace(4, 15, 500)
ax_scatter.plot(tt.f(y)*4 + x.min(), y, color="red")
fig_scatter.savefig("slides/img/t_as_tt.png", dpi=600)
plt.close(fig_scatter)

#%%

fig_bin, ax_bin = plt.subplots(figsize=(6, 4))
n, bins, patches = ax_bin.hist(t_as, 80)

ax_bin.set_yticks([])
ax_bin.set_xlabel(r"$t_a$ (h)")

def_color = patches[0].get_facecolor()

for b, p in zip(bins, patches):
    if b < b_high and b > b_low - p.get_width():
        p.set_facecolor(early_color)
    if b < g_high and b > g_low - p.get_width():
        p.set_facecolor(late_color)

x = np.linspace(6.5, 12.5, 300)
tt_line = ax_bin.plot(x, tt.f(x)*40, color=tt_color, linewidth=2, label="Travel time function")

labels = [tt_line[0].get_label(), "Early arrivals", "Late arrivals", "On time arrivals"]
handles = [tt_line[0], Patch(facecolor=early_color), Patch(facecolor=late_color), Patch(facecolor=def_color)]
ax_bin.legend(handles, labels)
ax_bin.set_xlim(6.5, 12.5)
fig_bin.savefig("slides/img/t_as_bins_tt.png", dpi=600)
plt.close(fig_bin)

#%%
h_len = .4
arc_len = .3
text_dist = .15
x = np.linspace(6.5, 12.5, 300)
fig_tt, ax_tt = plt.subplots(figsize=(7, 4))
ax_tt.plot(x, tt.f(x), linewidth=2, color=tt_color, label='Travel time')

ax_tt.set_yticks([])
ax_tt.set_xlabel(r"$t^*$ (h)")

tg_b = ax_tt.plot([bs[0], bs[1]], [tt.f(bs[0]), tt.f(bs[1])], color=early_color)
h_b =  ax_tt.plot([bs[0], bs[0] + h_len], [tt.f(bs[0])]*2, color=early_color)
b_arc = Arc([bs[0], tt.f(bs[0])], arc_len,
            arc_len*ax_tt.get_data_ratio()**(1/2), theta1=0,
            theta2=np.degrees(np.arctan(par[0])), color=early_color)
ax_tt.text(bs[0] + text_dist, tt.f(bs[0]) +
           text_dist*ax_tt.get_data_ratio()**(1/2),
           r"arctan$(\beta)$", size=8, color=early_color)
ax_tt.add_patch(b_arc)

tg_g = ax_tt.plot([gs[0], gs[1]], [tt.f(gs[0]), tt.f(gs[1])], color=late_color)
h_g =  ax_tt.plot([gs[1] - h_len, gs[1]], [tt.f(gs[1])]*2, color=late_color)
g_arc = Arc([gs[1], tt.f(gs[1])], arc_len,
            arc_len*ax_tt.get_data_ratio()**(1/2), angle=180, theta2=0,
            theta1=-np.degrees(np.arctan(par[1])), color=late_color)
ax_tt.text(gs[1] - text_dist, tt.f(gs[1]) +
           text_dist*ax_tt.get_data_ratio()**(1/2),
           r"arctan$(\gamma)$", size=8, color=late_color, ha='right')

ax_tt.add_patch(g_arc)

ax_tt.fill_between([bs[0], bs[1]], [ax_tt.get_ylim()[1]]*2, color=early_color, alpha=.2, label='Early arrival')
ax_tt.fill_between([gs[0], gs[1]], [ax_tt.get_ylim()[1]]*2, color=late_color, alpha=.2, label='Late arrival')

ax_tt.legend()

fig_tt.savefig("slides/img/tt_early_late.png", dpi=600)
plt.close(fig_tt)

#%%
_, _, _, points = generate_arrival(100000, tt, *par)
x = jnp.linspace(6, 13, 300)
ll = lambda x: likelihood(tt, x, *par)
lx = vmap(ll)(x)

#%%

fig_ll, ax_ll = plt.subplots(figsize=(6, 4))
ax_ll.hist(points, 400, label='Sampled points')
ax_ll.set_xlim(6, 12.5)
ax_ll.legend()
ax_ll.set_yticks([])
ax_ll.set_xlabel(r"$t_a$ (h)")
shade = ax_ll.fill_between(x, lx*4000, color="orange", alpha=.4, label=r'Likelihood of $t_a$')
shade.set_visible(False)
fig_ll.savefig("slides/img/hist_no_ll.png", dpi=600)
shade.set_visible(True)
ax_ll.legend()
fig_ll.savefig("slides/img/hist_ll.png", dpi=600)
plt.close(fig_ll)

#%%
num_ll = 1000
# par = (.4, 1.3, 9.5, .3, 1)
par[3] = .3
_, _, _, t_as_ll = generate_arrival(num_ll, tt, *par)

ll = lambda x, y: total_log_lik(tt, t_as_ll)(x, y, *par[2:])
betas_contour =jnp.linspace(.01, .99, 200)
gammas_contour = jnp.linspace(1.01, 4, 200)
matrix_ll = vmap(vmap(ll, (0, None)), (None, 0))(betas_contour, gammas_contour) # vmap(ll_actual)(*m_contour)

#%%

fig_ct, ax_ct = plt.subplots(figsize=(6, 4))
X, Y = jnp.meshgrid(betas_contour, gammas_contour)
ax_ct.contour(betas_contour, gammas_contour, matrix_ll, levels=50)
# ax_ct.plot(par[0], par[1], 'or')
ax_ct.set_xlabel(r"$\mu_\beta$")
ax_ct.set_ylabel(r"$\mu_\gamma$")
fig_ct.savefig("slides/img/contour_beautiful.png", dpi=600)

#%%
# par = (.4, 1.3, 9.5, .03, 1)
par[3]=.03
_, _, _, t_as_ll = generate_arrival(num_ll, tt, *par)

ll = lambda x, y: total_log_lik(tt, t_as_ll)(x, y, *par[2:])
betas_contour =jnp.linspace(.01, .99, 200)
gammas_contour = jnp.linspace(1.01, 4, 200)
matrix_ll = vmap(vmap(ll, (0, None)), (None, 0))(betas_contour, gammas_contour) # vmap(ll_actual)(*m_contour)

#%%

fig_ctb, ax_ctb = plt.subplots(figsize=(6, 4))
X, Y = jnp.meshgrid(betas_contour, gammas_contour)
ax_ctb.contour(betas_contour, gammas_contour, matrix_ll, levels=50)
ax_ctb.plot(par[0], par[1], 'or')
ax_ctb.set_xlabel(r"$\mu_\beta$")
ax_ctb.set_ylabel(r"$\mu_\gamma$")
fig_ctb.savefig("slides/img/contour_ugly.png", dpi=600)
