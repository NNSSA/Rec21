import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.style.use("../template.mplstyle")
# purple - green - darkgoldenrod - blue - red
colors = ["purple", "#306B37", "darkgoldenrod", "#3F7BB6", "#BF4145", "#cf630a"]

mean, std_dev, mean_wr, std_dev_wr = (
    -14.241866,
    8.406500335592806,
    -2.7567148e-07,
    1.5758628076117043,
)

rho_x0 = np.load("x0.npy") * std_dev_wr + mean_wr
rho_x1 = np.load("x1.npy") * std_dev + mean
rho_sample = np.load("timeline_4steps_step_500.npy") * std_dev + mean

mean_x1, std_x1 = np.mean(rho_x1), np.std(rho_x1)

#################

plt.figure(figsize=(8.7, 7))
ax = plt.subplot(111)
ax.tick_params(axis="x", which="major", pad=6)
plt.tick_params(axis="both", which="major", labelsize=25)
plt.tick_params(axis="both", which="minor", labelsize=25)

face_alpha = 0.3
edge_alpha = 1.0
face_color = mcolors.to_rgba(colors[2], alpha=face_alpha)
edge_color = mcolors.to_rgba(colors[2], alpha=edge_alpha)
n, bins, patches = ax.hist(
    (rho_x1.flatten() - mean_x1) / std_x1,
    density=True,
    histtype="stepfilled",
    bins=60,
    color=face_color,
    edgecolor=edge_color,
    lw=2.2,
    label=r"$\mathrm{Ground\ Truth}$",
    zorder=-3,
)
for patch in patches:
    patch.set_facecolor(face_color)
    patch.set_edgecolor(edge_color)
    patch.set_linewidth(2.2)

#

face_alpha = 0.3
edge_alpha = 1.0
face_color = mcolors.to_rgba(colors[1], alpha=face_alpha)
edge_color = mcolors.to_rgba(colors[1], alpha=edge_alpha)
n, bins, patches = ax.hist(
    (rho_x0.flatten() - mean_x1) / std_x1,
    density=True,
    histtype="stepfilled",
    bins=60,
    color=face_color,
    edgecolor=edge_color,
    lw=2.2,
    label=r"$\mathrm{Wedge\ Filtered}$",
    zorder=-4,
)
for patch in patches:
    patch.set_facecolor(face_color)
    patch.set_edgecolor(edge_color)
    patch.set_linewidth(2.2)

#

face_alpha = 0.3
edge_alpha = 1.0
face_color = mcolors.to_rgba(colors[3], alpha=face_alpha)
edge_color = mcolors.to_rgba(colors[3], alpha=edge_alpha)
n, bins, patches = ax.hist(
    (rho_sample.flatten() - mean_x1) / std_x1,
    density=True,
    histtype="stepfilled",
    bins=60,
    color=face_color,
    edgecolor=edge_color,
    lw=2.2,
    label=r"$\mathrm{Reconstructed\ Sample}$",
    zorder=-2,
)
for patch in patches:
    patch.set_facecolor(face_color)
    patch.set_edgecolor(edge_color)
    patch.set_linewidth(2.2)

# plotting specifications

plt.axis(xmin=-3, xmax=3)
plt.legend(
    loc="upper left",
    frameon=False,
    markerfirst=True,
    prop={"size": 19},
    handlelength=1.65,
    handletextpad=0.7,
    numpoints=1,
)
plt.xlabel(r"$\mathrm{Standardized\ voxel\ values}$", fontsize=28)
plt.ylabel(r"$\mathrm{Distribution}$", fontsize=28, labelpad=8)
plt.savefig("Distributions.pdf")
