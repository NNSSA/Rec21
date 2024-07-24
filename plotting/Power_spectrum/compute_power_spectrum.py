import numpy as np
import os
from tqdm import tqdm

mean, std_dev, mean_wr, std_dev_wr = (
    -14.241866,
    8.406500335592806,
    -2.7567148e-07,
    1.5758628076117043,
)


def PS_1D(rho, BOX_LEN=512, HII_DIM=128, N_BINS=10):
    rho = rho - rho.mean()
    rho_fourier = np.fft.fftn(rho)
    power_spectrum = (np.abs(rho_fourier)) ** 2

    k_x = np.fft.fftfreq(HII_DIM, BOX_LEN / HII_DIM) * 2 * np.pi
    k_y = k_x
    k_z = k_x

    k_magnitude = np.sqrt(
        k_x[:, None, None] ** 2 + k_y[None, :, None] ** 2 + k_z[None, None, :] ** 2
    )

    num_k_bins = N_BINS
    k_bins = np.geomspace(2.0 * np.pi / BOX_LEN, k_magnitude.max(), num_k_bins)
    power_spectrum_binned = np.histogram(
        k_magnitude, bins=k_bins, weights=power_spectrum
    )[0]
    count_in_bins = np.histogram(k_magnitude, bins=k_bins)[0]
    power_spectrum_binned /= count_in_bins
    k_bins = (k_bins[1:] + k_bins[:-1]) / 2
    return power_spectrum_binned * (k_bins**3) / (2 * np.pi**2), k_bins


def compute_and_save_power_spectra(
    folder1, folder2, extra_files, output_file, box_size=512, d=128
):
    file_count = 1000
    total_files = file_count * 2 + len(extra_files)
    power_spectra = []

    with tqdm(total=total_files, desc="Computing power spectra", unit="file") as pbar:
        # Process first folder
        for i in range(file_count):
            file_path = os.path.join(folder1, f"sample_{i}.npy")
            rho = np.load(file_path) * std_dev + mean
            power_spectrum, k_bins = PS_1D(rho)
            power_spectra.append(power_spectrum)
            pbar.update(1)

        # Process second folder
        for i in range(file_count):
            file_path = os.path.join(folder2, f"sample_{i}.npy")
            rho = np.load(file_path) * std_dev + mean
            power_spectrum, k_bins = PS_1D(rho)
            power_spectra.append(power_spectrum)
            pbar.update(1)

        # Process extra files
        for file_path in extra_files:
            if "x0.npy" in file_path:
                rho = np.load(file_path) * std_dev_wr + mean_wr
            else:
                rho = np.load(file_path) * std_dev + mean
            power_spectrum, k_bins = PS_1D(rho)
            power_spectra.append(power_spectrum)
            pbar.update(1)

    # Save all power spectra with the wavenumber array
    power_spectra = np.array(power_spectra)
    np.savez(output_file, k_bins=k_bins, power_spectra=power_spectra)


folder1 = "../Data/samples_Rec21"
folder2 = "../Data/samples_Rec21_part2"
extra_files = [
    "../Data/samples_Rec21/x0.npy",
    "../Data/samples_Rec21/x1.npy",
    "../Data/average_samples.npy",
]
output_file = "power_spectra_logarithmic.npz"

compute_and_save_power_spectra(folder1, folder2, extra_files, output_file)
