import numpy as np
import os
import glob
import h5py

# Constants
HII_DIM = 128
BOX_LEN = 512  # Mpc
N_cubes = 10
redshift = 10.0
degrees = 65.0
psi = degrees / 180.0 * np.pi


# Rotate box along LoS
def rotations4(polycube, axes=(1, 2)):
    for i in range(4):
        yield np.rot90(polycube, i, axes)


# Generate wedge mask
def generate_mask():
    mask = np.ones((HII_DIM, HII_DIM, HII_DIM))
    freq = np.fft.fftfreq(HII_DIM, BOX_LEN / HII_DIM)
    i, j, k = np.indices((HII_DIM, HII_DIM, HII_DIM))
    zero_idx = (
        np.abs(freq[i]) < np.sqrt(freq[j] ** 2 + freq[k] ** 2) * np.tan(psi)
    ) | (np.abs(freq[i]) * 2 * np.pi < 0.05)
    mask[zero_idx] = 0
    return mask


def apply_mask_one_box(box, mask):
    box = np.fft.fftn(box)
    box = np.einsum("ijk, ijk -> ijk", box, mask)
    box = np.fft.ifftn(box)
    return np.real(box)


def process_files_in_chunks(directory, max_idx):
    files = glob.glob(os.path.join(directory, "run_*.npy"))
    for file in files[: max_idx + 1]:
        data = np.load(file)
        for box in data:
            box = np.transpose(box, (2, 0, 1))
            yield box


def save_directly_to_hdf5(output_filename, generator, mask):
    with h5py.File(output_filename, "w", libver="latest") as h5f:
        dset_rot = h5f.create_dataset(
            "T21_data",
            shape=(1, HII_DIM, HII_DIM, HII_DIM),
            maxshape=(None, HII_DIM, HII_DIM, HII_DIM),
            chunks=(1, HII_DIM, HII_DIM, HII_DIM),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        dset_masked = h5f.create_dataset(
            "T21_wr_data",
            shape=(1, HII_DIM, HII_DIM, HII_DIM),
            maxshape=(None, HII_DIM, HII_DIM, HII_DIM),
            chunks=(1, HII_DIM, HII_DIM, HII_DIM),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        idx_rot = 0
        idx_mask = 0
        for box in generator:
            rotated_boxes = list(rotations4(box))
            dset_rot.resize(idx_rot + len(rotated_boxes), axis=0)
            dset_rot[idx_rot : idx_rot + len(rotated_boxes)] = rotated_boxes
            idx_rot += len(rotated_boxes)

            masked_boxes = [
                apply_mask_one_box(rotated, mask) for rotated in rotated_boxes
            ]
            dset_masked.resize(idx_mask + len(masked_boxes), axis=0)
            dset_masked[idx_mask : idx_mask + len(masked_boxes)] = masked_boxes
            idx_mask += len(masked_boxes)


N_files = 2500
directory = "./data_T21s_LC_z=10.0_512Mpc_128Cells_10boxes"
output_file = "./data_T21s_LC_z={:.1f}_{:d}Mpc_{:d}Cells_{:d}boxes.h5".format(
    redshift, BOX_LEN, HII_DIM, N_files * 40
)
generator = process_files_in_chunks(directory, N_files)
wedge_mask = generate_mask()
save_directly_to_hdf5(output_file, generator, wedge_mask)
