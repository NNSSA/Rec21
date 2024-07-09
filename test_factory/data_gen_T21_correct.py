import numpy as np
import matplotlib.pyplot as plt
import py21cmfast as p21c
import os
import h5py
from shutil import rmtree
import random
from py21cmfast import global_params
from py21cmfast import plotting

HII_DIM = 32
BOX_LEN = 128
N_cubes = 1
redshift = 15.

psi = 50. / 180. * np.pi  # variable that defines the wedge


def remove_wedge(image: np.array):
    # For N_x = N_y = N_z

    f1 = np.fft.fftn(image)
    N = len(image[0])
    freq = np.fft.fftfreq(N)

    def rule_remove_wedge(i, j, k):
        return (
            0 if np.abs(freq[i]) < np.sqrt(freq[j]**2 + freq[k]**2) * np.tan(psi) else f1[i, j, k]
        )

    rule_vectorized = np.vectorize(rule_remove_wedge)
    shape = image.shape
    indices = [
        np.arange(shape[0])[:, np.newaxis, np.newaxis],
        np.arange(shape[1])[:, np.newaxis],
        np.arange(shape[2]),
    ]

    f2 = rule_vectorized(*indices)

    return np.real(np.fft.ifftn(f2))

user_params = {
    "HII_DIM": HII_DIM,
    "BOX_LEN": BOX_LEN,
    "USE_FFTW_WISDOM": True,
    "USE_INTERPOLATION_TABLES": True,
    "FAST_FCOLL_TABLES": True,
    "USE_RELATIVE_VELOCITIES": True,
    "POWER_SPECTRUM": 5,
}

flag_options = {
    "INHOMO_RECO": True,
    "USE_MASS_DEPENDENT_ZETA": True,
    "USE_TS_FLUCT": True,
    "USE_MINI_HALOS": False,
    "FIX_VCB_AVG": False,
}

if not os.path.isdir("_cache"):
    os.mkdir("_cache")
p21c.config["direc"] = "_cache"
p21c.cache_tools.clear_cache(direc="_cache")

vcb_boxes = []
T21_boxes = []


def generate_cubes(num_cubes: int):
    # For when we want to change astro params
    # astro_params = {
    #     "ALPHA_ESC": -0.3,
    #     "F_ESC10": -1.2,
    #     "ALPHA_STAR": 0.5,
    #     "F_STAR10": -1.5,
    #     "t_STAR": 0.5,
    #     "F_STAR7_MINI": -1.75,
    #     "ALPHA_STAR_MINI": 0,
    #     "F_ESC7_MINI": -2.25,
    #     "L_X": 40.5,
    #     "L_X_MINI": 40.5,
    #     "NU_X_THRESH": 200.0,
    #     "A_VCB": 1.0,
    #     "A_LW": 2.0,
    #     "R_BUBBLE_MAX": 20.0,
    # }

    for i in range(num_cubes):
        # Set seed for random initial conditions
        random_seed = random.randint(10, 1000000)

        # Keep initial conditions separate from run_coeval in case we want to change cosmo parameters
        initial_conditions = p21c.initial_conditions(
            user_params=user_params, random_seed=random_seed, direc="_cache"
        )

        # Run coeval box at single redshift
        box_coeval = p21c.run_coeval(
            redshift=redshift,
            init_box=initial_conditions,
            user_params=user_params,
            flag_options=flag_options,
            cosmo_params=p21c.CosmoParams(SIGMA_8=0.8),
            astro_params=p21c.AstroParams({"HII_EFF_FACTOR": 20.0, "R_BUBBLE_MAX": 40.0}),
            # random_seed=random_seed,  # Don't think we need to pass this one again, since it's already in initial_conditions
            direc="_cache",
            write=False,
        )

        vcb_boxes.append(initial_conditions.lowres_vcb)
        T21_boxes.append(box_coeval.brightness_temp)

        print("Fields done for i = %d" % (i + 1))
        p21c.cache_tools.clear_cache(direc="_cache")

    return np.array(vcb_boxes), np.array(T21_boxes)


vcbs, T21s = generate_cubes(num_cubes=N_cubes)
T21s_wr = np.array([remove_wedge(image) for image in T21s], dtype=np.float32)

# plt.figure()
# plt.subplot(121)
# plt.imshow(T21s[0][0,:,:])
# plt.subplot(122)
# plt.imshow(T21s_wr[0][0,:,:])
# plt.show()

# np.save(
#     "vcbs_z={:.1f}_{:d}Mpc_{:d}Cells_{:d}boxes_norelvel_psi={:.1f}.npy".format(
#         redshift, BOX_LEN, HII_DIM, N_cubes, psi
#     ),
#     vcbs,
# )
np.save(
    "T21s_z={:.1f}_{:d}Mpc_{:d}Cells_{:d}boxes_norelvel_psi={:.1f}.npy".format(
        redshift, BOX_LEN, HII_DIM, N_cubes, psi
    ),
    T21s,
)
# np.save(
#     "T21s_wr_z={:.1f}_{:d}Mpc_{:d}Cells_{:d}boxes_norelvel_psi={:.1f}.npy".format(
#         redshift, BOX_LEN, HII_DIM, N_cubes, psi
#     ),
#     T21s_wr,
# )

