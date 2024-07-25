import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../template.mplstyle")
# purple - green - darkgoldenrod - blue - red
colors = ["purple", "#306B37", "darkgoldenrod", "#3F7BB6", "#BF4145", "#cf630a"]

fig = plt.figure(figsize=(6, 12.5))

gs = fig.add_gridspec(4, 2, height_ratios=[2, 0.26, 1, 1], hspace=0.08, wspace=0.08)

ax1 = fig.add_subplot(gs[0, :])
ax1.tick_params(axis="x", which="major", pad=6)
ax1.tick_params(axis="both", which="major", labelsize=21)
ax1.tick_params(axis="both", which="minor", labelsize=21)

kperp = np.geomspace(1e-2, 1, 1000)
kpar = np.tan(65.0 / 180 * np.pi) * kperp

ax1.plot(kperp, kpar, color=colors[3], lw=1.6)
ax1.fill_between(
    kperp,
    kpar,
    where=(kperp < kpar),
    interpolate=True,
    color=colors[3],
    alpha=0.2,
)
for i in range(1, 30):
    ax1.plot(
        kperp,
        kpar * (0.8) ** i,
        color=colors[3],
        lw=0.7,
        alpha=0.5,
    )

ax1.plot(kperp, np.repeat(0.05, len(kperp)), color=colors[1], lw=1.6, zorder=-10)
ax1.fill_between(
    kperp,
    np.repeat(0.05, len(kperp)),
    interpolate=True,
    color=colors[1],
    alpha=0.2,
    zorder=-10,
)

ax1.set_xlim(0.01, 1.0)
ax1.set_ylim(0.01, 1.0)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"$k_{{\perp}}\ [\mathrm{Mpc}^{{-1}}]$", fontsize=22, labelpad=3)
ax1.set_ylabel(r"$k_{{\parallel}}\ [\mathrm{Mpc}^{{-1}}]$", fontsize=22)
ax1.text(
    0.5,
    0.175,
    r"$\mathrm{Intrinsic\ Foregrounds}$",
    ha="center",
    va="center",
    transform=ax1.transAxes,
    color=colors[1],
    zorder=10,
    fontsize=20,
)
ax1.text(
    0.70,
    0.55,
    r"$\mathrm{Foreground}$",
    color=colors[3],
    ha="center",
    va="center",
    transform=ax1.transAxes,
    fontsize=20,
)
ax1.text(
    0.70,
    0.484,
    r"$\mathrm{Wedge}$",
    color=colors[3],
    ha="center",
    va="center",
    transform=ax1.transAxes,
    fontsize=20,
)
ax1.text(
    0.26,
    0.76,
    r"$\mathrm{Unfiltered}$",
    ha="center",
    va="center",
    transform=ax1.transAxes,
    fontsize=20,
)
ax1.text(
    0.26,
    0.694,
    r"$\mathrm{Modes}$",
    ha="center",
    va="center",
    transform=ax1.transAxes,
    fontsize=20,
)

# Create a 2x2 grid of subplots for the images
ax2_1 = fig.add_subplot(gs[2, 0])
ax2_2 = fig.add_subplot(gs[2, 1])
ax2_3 = fig.add_subplot(gs[3, 0])
ax2_4 = fig.add_subplot(gs[3, 1])

rho_x1 = np.load("timeline_x1.npy")
rho_x0 = np.load("timeline_x0.npy")

ax2_1.imshow(rho_x1[10, :, :], cmap="viridis")
ax2_2.imshow(rho_x0[10, :, :], cmap="viridis")
ax2_3.imshow(rho_x1[::-1, :, 10], cmap="viridis")
ax2_4.imshow(rho_x0[::-1, :, 10], cmap="viridis")

ax2_3.annotate(
    "",
    xy=(10, 60),
    xytext=(10, 120),
    transform=ax2_3.transAxes,
    arrowprops=dict(
        facecolor="black", edgecolor="black", headwidth=3, headlength=4, width=0.01
    ),
)
ax2_3.text(
    13,
    115,
    r"$\mathrm{Higher\ redshift}$",
    color="black",
    rotation=90,
    fontsize=9,
)

ax2_1.set_xticks([])
ax2_1.set_yticks([])
ax2_2.set_xticks([])
ax2_2.set_yticks([])
ax2_3.set_xticks([])
ax2_3.set_yticks([])
ax2_4.set_xticks([])
ax2_4.set_yticks([])

ax2_3.set_xlabel(r"$\mathrm{Ground\ Truth}$", fontsize=21)
ax2_4.set_xlabel(r"$\mathrm{Wedge\ Filtered}$", fontsize=21)

ax2_3.set_ylabel(r"$\mathrm{Line\ of\ Sight}$", fontsize=21)
ax2_1.set_ylabel(r"$\mathrm{Transverse}$", fontsize=21)

ax2_1.set_aspect("equal")
ax2_2.set_aspect("equal")
ax2_3.set_aspect("equal")
ax2_4.set_aspect("equal")

plt.savefig("Wedge_Plots.pdf")
