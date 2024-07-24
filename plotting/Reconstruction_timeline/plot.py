import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../template.mplstyle")

# Data for plotting
x0 = np.load("timeline_x0_4steps.npy")
x1 = np.load("timeline_x1_4steps.npy")
step1 = np.load("timeline_4steps_step_125.npy")
step2 = np.load("timeline_4steps_step_250.npy")
step3 = np.load("timeline_4steps_step_375.npy")
step4 = np.load("timeline_4steps_step_500.npy")
average = np.load("average_2000_samples.npy")

# Create the plot
fig, ax = plt.subplots(nrows=2, ncols=7, figsize=(16, 7))

ax[0, 0].set_title(r"$t = 0\ (x_{21}^\mathrm{wf})$", pad=8, fontsize=21)
ax[0, 0].imshow(x0[10, :, :], cmap="viridis")
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_ylabel(r"$\mathrm{Transverse}$", labelpad=6, fontsize=23)

ax[0, 1].set_title(r"$t = 0.25$", pad=8, fontsize=21)
ax[0, 1].imshow(step1[10, :, :], cmap="viridis")
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])

ax[0, 2].set_title(r"$t = 0.5$", pad=8, fontsize=21)
ax[0, 2].imshow(step2[10, :, :], cmap="viridis")
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])

ax[0, 3].set_title(r"$t = 0.75$", pad=8, fontsize=21)
ax[0, 3].imshow(step3[10, :, :], cmap="viridis")
ax[0, 3].set_xticks([])
ax[0, 3].set_yticks([])

ax[0, 4].set_title(r"$t = 1$", pad=8, fontsize=21)
ax[0, 4].imshow(step4[10, :, :], cmap="viridis")
ax[0, 4].set_xticks([])
ax[0, 4].set_yticks([])

ax[0, 5].set_title(r"$\mathrm{Ground\ Truth}$", pad=8, fontsize=21)
ax[0, 5].imshow(x1[10, :, :], cmap="viridis")
ax[0, 5].set_xticks([])
ax[0, 5].set_yticks([])

ax[0, 6].set_title(r"$\mathrm{Average}$", pad=8, fontsize=21)
ax[0, 6].imshow(average[10, :, :], cmap="viridis")
ax[0, 6].set_xticks([])
ax[0, 6].set_yticks([])


ax[1, 0].imshow(x0[:, :, 10], cmap="viridis")
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 0].set_ylabel(r"$\mathrm{Line\,of\,sight}$", labelpad=6, fontsize=23)

ax[1, 1].imshow(step1[:, :, 10], cmap="viridis")
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])

ax[1, 2].imshow(step2[:, :, 10], cmap="viridis")
ax[1, 2].set_xticks([])
ax[1, 2].set_yticks([])

ax[1, 3].imshow(step3[:, :, 10], cmap="viridis")
ax[1, 3].set_xticks([])
ax[1, 3].set_yticks([])

ax[1, 4].imshow(step4[:, :, 10], cmap="viridis")
ax[1, 4].set_xticks([])
ax[1, 4].set_yticks([])

ax[1, 5].imshow(x1[:, :, 10], cmap="viridis")
ax[1, 5].set_xticks([])
ax[1, 5].set_yticks([])

ax[1, 6].imshow(average[:, :, 10], cmap="viridis")
ax[1, 6].set_xticks([])
ax[1, 6].set_yticks([])

plt.tight_layout()
plt.subplots_adjust(hspace=-0.495)
plt.savefig("Reconstruction_timeline.pdf")
