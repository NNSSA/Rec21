import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("../template.mplstyle")
# purple - green - darkgoldenrod - blue - red
colors = ["purple", "#306B37", "darkgoldenrod", "#3F7BB6", "#BF4145", "#cf630a"]


def ctr_level(hist, lvl, infinite=False):
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist = cum_hist / cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)
    clist = [0] + [hist[-i] for i in alvl]
    if not infinite:
        return clist[1:]
    return clist


###


def get_hist(data, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(data))
    hist, bin_edges = np.histogram(data, bins=num_bins, weights=weights)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return hist, bin_edges, bin_centres


###


def plot_hist(data, num_bins=30, weights=[None], color=None):
    if not any(weights):
        weights = np.ones(len(data))
    if color == None:
        color = "darkblue"

    hist, bin_edges, bin_centres = get_hist(data, num_bins=num_bins, weights=weights)
    plt.plot(bin_centres, hist / max(hist), color=color, lw=2)
    plt.step(bin_centres, hist / max(hist), where="mid", color=color)
    plt.show()


###


def limits():
    lower = []
    upper = []

    for index in range(len(k_bins)):
        hist, _, bin_centres = get_hist(
            power_spectra_samples[:, index] / ps_x1[index], num_bins=20
        )
        xarray = np.linspace(min(bin_centres), max(bin_centres), 100)
        interpolator = PchipInterpolator(bin_centres, hist)(xarray)
        levels = ctr_level(interpolator.copy(), [0.95])
        pos = [
            np.searchsorted(interpolator[: np.argmax(interpolator)], levels)[0] - 1,
            -np.searchsorted(
                (interpolator[::-1])[: np.argmax(interpolator[::-1])], levels
            )[0],
        ]
        lower.append(min(xarray[pos[0]], xarray[pos[1]]))
        upper.append(max(xarray[pos[0]], xarray[pos[1]]))
        # print(min(xarray[pos[0]], xarray[pos[1]]), max(xarray[pos[0]], xarray[pos[1]]))
        # plt.axvline(min(xarray[pos[0]], xarray[pos[1]]), c="red")
        # plt.axvline(max(xarray[pos[0]], xarray[pos[1]]), c="red")
        # plot_hist(power_spectra_samples[:, index] / ps_x1[index], num_bins=20)

    return np.array(lower), np.array(upper)


# Make the 2D PS plot


def power_spectrum_2d(box, HII_DIM=128, BOX_LEN=512, num_k_bins=50):
    # Calculate the 3D Fourier transform
    X = np.fft.fftn(box)
    X = np.abs(X) ** 2

    # Calculate the frequency components
    freq = np.fft.fftfreq(HII_DIM, BOX_LEN / HII_DIM) * 2 * np.pi

    # Create meshgrids for k_x, k_y, and k_z
    k_x, k_y, k_z = np.meshgrid(freq, freq, freq, indexing="ij")

    # Calculate the magnitude of the perpendicular component (k_perp)
    k_perp = np.sqrt(k_y**2 + k_z**2)

    # Flatten arrays for histogram calculation
    k_x = k_x.flatten()
    k_perp = k_perp.flatten()
    X = X.flatten()

    # Bin edges
    k_x_bins = np.geomspace(2.0 * np.pi / BOX_LEN, k_x.max(), num_k_bins)
    k_perp_bins = np.geomspace(2.0 * np.pi / BOX_LEN, k_perp.max(), num_k_bins)

    # 2D histogram to calculate the binned power spectrum
    power_spectrum_binned = np.histogram2d(
        k_x, k_perp, bins=(k_x_bins, k_perp_bins), weights=X
    )[0]
    count_in_bins = np.histogram2d(k_x, k_perp, bins=(k_x_bins, k_perp_bins))[0]

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        power_spectrum_binned = np.nan_to_num(power_spectrum_binned / count_in_bins)

    k_x = k_x_bins[1:]
    k_perp = k_perp_bins[1:]

    return (
        power_spectrum_binned
        * ((k_x[:, None] ** 3) + (k_perp[None, :] ** 3))
        / (2 * np.pi**2),
        k_x,
        k_perp,
    )


rho_x1 = np.load("timeline_x1.npy")
rho_sample = np.load("timeline_step_500.npy")

ps_x1, k_parallel_x, k_perp_x1 = power_spectrum_2d(rho_x1, num_k_bins=10)
ps_sample, k_parallel_sample, k_perp_sample = power_spectrum_2d(
    rho_sample, num_k_bins=10
)

ratio = np.nan_to_num(ps_sample / ps_x1)

plt.figure(figsize=(22, 8))
ax1 = plt.subplot(121)
ax1.tick_params(axis="x", which="major", pad=6)
ax1.tick_params(axis="both", which="major", labelsize=28)
ax1.tick_params(axis="both", which="minor", labelsize=28)

extent = [k_perp_x1[0], k_perp_x1[-1], k_parallel_x[0], k_parallel_x[-1]]
img = ax1.pcolormesh(
    k_perp_x1, k_parallel_x, ratio, cmap="Blues", vmin=0.4, vmax=1.6, alpha=0.85
)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.15)

# Plot the colorbar
cbar = plt.colorbar(img, cax=cax, orientation="vertical")
cbar.ax.set_ylabel(
    r"$\Delta_{21}^2 / \Delta_{21,\mathrm{gr.\, truth}}^2$",
    fontsize=28,
    rotation=270,
    labelpad=44,
)

x_array = np.geomspace(k_perp_x1.min(), k_perp_x1.max(), 1000)
ax1.plot(
    x_array,
    np.maximum(x_array * np.tan(65 / 180 * np.pi), 0.05),
    color="black",
    ls=(0, (4, 2.5)),
    lw=2.2,
    zorder=10,
)
ax1.text(
    0.0425,
    0.120,
    r"$\mathrm{Unfiltered\ modes}$",
    color="black",
    rotation=45,
    fontsize=28,
    fontweight="bold",
)

ax1.annotate(
    "",
    xy=(0.125, 0.435),
    xytext=(0.155, 0.35),
    arrowprops=dict(
        facecolor="black", edgecolor="black", headwidth=9, headlength=7, width=0.1
    ),
)

ax1.semilogx()
ax1.semilogy()
ax1.set_xlim(k_perp_x1[0], k_perp_x1[-1])
ax1.set_ylim(k_parallel_x[0], k_parallel_x[-1])
ax1.set_ylabel(r"$k_{\parallel}\ [\mathrm{Mpc}^{-1}]$", fontsize=31)
ax1.set_xlabel(r"$k_{\perp}\ [\mathrm{Mpc}^{-1}]$", fontsize=31)

# Make the 1D PS plot

data = np.load("power_spectra_logarithmic.npz")
k_bins = data["k_bins"]
power_spectra = data["power_spectra"] * k_bins**3 / (2 * np.pi**2)
power_spectra_samples = power_spectra[:-3]

ps_x0 = power_spectra[-3]
ps_x1 = power_spectra[-2]
ps_avg = power_spectra[-1]

ps_min, ps_max = limits()

ax2 = plt.subplot(122)
ax2.tick_params(axis="x", which="major", pad=6)
plt.tick_params(axis="both", which="major", labelsize=28)
plt.tick_params(axis="both", which="minor", labelsize=28)

ax2.plot(
    k_bins,
    ps_x1 / ps_x1,
    color="black",
    ls=(1.6, (2, 1, 2, 1)),
    lw=2.5,
    label=r"$\mathrm{Ground\, Truth}$",
)
ax2.plot(
    k_bins,
    ps_x0 / ps_x1,
    color=colors[2],
    lw=2.5,
    ls=(0, (1, 1.05)),
    label=r"$\mathrm{Wedge\, Filtered}$",
)

ax2.fill_between(
    np.append(k_bins, k_bins[::-1]),
    np.append(ps_min, ps_max[::-1]),
    color=colors[3],
    alpha=0.5,
    lw=0,
    zorder=0,
)
ax2.plot(
    k_bins,
    ps_min,
    color=colors[3],
    lw=2.5,
    zorder=2,
    label=r"$\mathrm{Reconstructed}\ (95\%\ \mathrm{CL})$",
)
ax2.plot(k_bins, ps_max, color=colors[3], lw=2.5, zorder=2)
ax2.plot(
    k_bins,
    ps_avg / ps_x1,
    color=colors[1],
    lw=2.5,
    ls=(0, (3, 1, 1, 1)),
    label=r"$\mathrm{Averaged\, Lightcone}$",
)

# plotting specifications

ax2.semilogx()
plt.axis(xmin=min(k_bins), xmax=max(k_bins) * 1.01, ymin=0.5, ymax=1.5)
ax2.legend(
    loc="upper left",
    frameon=False,
    markerfirst=True,
    prop={"size": 22},
    handlelength=1.65,
    handletextpad=0.7,
    numpoints=1,
)
ax2.set_xlabel(r"$k\ [\mathrm{Mpc}^{-1}]$", fontsize=31)
ax2.set_ylabel(
    r"$\Delta_{21}^2 / \Delta_{21,\mathrm{gr.\, truth}}^2$", fontsize=31, labelpad=8
)
plt.subplots_adjust(wspace=0.35)
for axis in ["top", "bottom", "left", "right"]:
    for ax in [ax1, ax2]:
        ax.spines[axis].set_linewidth(2.5)

plt.savefig("Power_spectra.pdf")
