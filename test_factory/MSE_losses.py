import numpy as np
import os
from tqdm import tqdm

mean, std_dev, mean_wr, std_dev_wr = (
    -14.241866,
    8.406500335592806,
    -2.7567148e-07,
    1.5758628076117043,
)

rho_x0 = (
    np.load("../Data/samples_Rec21_200k_finalfinal_5epochs_halfdata_48M_samples/x0.npy")
    * std_dev_wr
    + mean_wr
)
rho_x1 = (
    np.load("../Data/samples_Rec21_200k_finalfinal_5epochs_halfdata_48M_samples/x1.npy")
    * std_dev
    + mean
)

mean_x1, std_x1 = np.mean(rho_x1), np.std(rho_x1)


def compute_MSE(folder1, folder2):
    file_count1 = len([f for f in os.listdir(folder1) if f.startswith("sample_")])
    file_count2 = len([f for f in os.listdir(folder2) if f.startswith("sample_")])
    total_files = file_count1 + file_count2
    MSE = 0

    # Create a tqdm progress bar
    with tqdm(total=total_files, desc="Computing power spectra", unit="file") as pbar:
        # Process first folder
        for i in range(file_count1):
            file_path = os.path.join(folder1, f"sample_{i}.npy")
            rho = np.load(file_path) * std_dev + mean
            mse = np.mean((rho - rho_x1) ** 2)
            MSE += mse
            pbar.update(1)  # Update progress bar

        # Process second folder
        for i in range(file_count2):
            file_path = os.path.join(folder2, f"sample_{i}.npy")
            rho = np.load(file_path) * std_dev + mean
            mse = np.mean((rho - rho_x1) ** 2)
            MSE += mse
            pbar.update(1)  # Update progress bar

    return MSE / total_files


# Usage
folder1 = "../Data/samples_Rec21_200k_finalfinal_5epochs_halfdata_48M_samples"
folder2 = "../Data/samples_Rec21_200k_finalfinal_5epochs_halfdata_48M_samples_part2"

MSE = compute_MSE(folder1, folder2)
print("Identity MSE between x0 and x1:", np.mean((rho_x0 - rho_x1) ** 2))
print("Variane of x1:", np.var(rho_x1))
print(f"MSE of reconstructed samples: {MSE}")
print(f"RMSE of reconstructed samples: {np.sqrt(MSE)}")


# Compute histogram of rho_x1 and a sample from folder1
import matplotlib.pyplot as plt

plt.hist((rho_x0.flatten() - mean_x1) / std_x1, bins=100, alpha=0.5, label="x0")
plt.hist((rho_x1.flatten() - mean_x1) / std_x1, bins=100, alpha=0.5, label="x1")
rho = np.load(os.path.join(folder1, "sample_10.npy")) * std_dev + mean
plt.hist((rho.flatten() - mean_x1) / std_x1, bins=100, alpha=0.5, label="sample_0")
plt.legend()
plt.show()
