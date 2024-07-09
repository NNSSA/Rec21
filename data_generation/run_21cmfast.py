import py21cmfast as p21c
import os
import numpy as np
import uuid
import hashlib
import fasteners

# Setting up run parameters

HII_DIM = 128
BOX_LEN = 512  # Mpc
N_cubes = 10
redshift = 10.0


user_params = {
    "HII_DIM": HII_DIM,
    "BOX_LEN": BOX_LEN,
    "USE_FFTW_WISDOM": True,
    "USE_INTERPOLATION_TABLES": True,
    "FAST_FCOLL_TABLES": True,
    "USE_RELATIVE_VELOCITIES": False,
    "POWER_SPECTRUM": 5,
    # "N_THREADS": 48
}


flag_options = {
    "INHOMO_RECO": True,
    "USE_MASS_DEPENDENT_ZETA": True,
    "USE_TS_FLUCT": True,
    "USE_MINI_HALOS": False,
}

if not os.path.isdir("_cache"):
    os.mkdir("_cache")
p21c.config["direc"] = "_cache"


cosmoparams = p21c.CosmoParams(
    hlittle=0.6727, OMm=0.3166, OMb=0.0494, POWER_INDEX=0.9649, SIGMA_8=0.8120
)  # sets the CosmoParams to Planck18


dz = p21c.global_params.ZPRIME_STEP_FACTOR  # step as (z1 + 1) = dz*(z0 + 1)


def scan_for_redshifts(zcenter=10.0, BOX_LEN=BOX_LEN, tol=1.0e-06):
    L0 = cosmoparams.cosmo.comoving_distance(10).value

    z0 = 0.0
    z1 = zcenter
    zmin = (z0 + z1) / 2.0

    while (
        np.abs(cosmoparams.cosmo.comoving_distance(zmin).value - (L0 - BOX_LEN // 2))
        > tol
    ):
        if cosmoparams.cosmo.comoving_distance(zmin).value > (L0 - BOX_LEN // 2):
            z1 = zmin
        else:
            z0 = zmin
        zmin = (z1 + z0) / 2.0

    z0 = zcenter
    z1 = 15.0
    zmax = (z0 + z1) / 2.0

    while (
        np.abs(cosmoparams.cosmo.comoving_distance(zmax).value - (L0 + BOX_LEN // 2))
        > tol
    ):
        if cosmoparams.cosmo.comoving_distance(zmax).value > (L0 + BOX_LEN // 2):
            z1 = zmax
        else:
            z0 = zmax
        zmax = (z1 + z0) / 2.0
    return round(zmin, 6), round(zmax, 6)


def chop_box(box):
    return box[:, :, :HII_DIM]


zmin, zmax = scan_for_redshifts()


def get_seed():
    seed_str = str(uuid.uuid4())
    hash_seed = hashlib.sha256(seed_str.encode()).hexdigest()
    return int(hash_seed, 16) % (2**32)


def save_parameters_to_txt(params, filename="data/parameters.txt"):
    lock = fasteners.InterProcessLock(f"{filename}.lock")
    with lock:
        with open(filename, "a") as file:
            file.write(" ".join(map(str, params)) + "\n")


def generate_lightcone():
    random_seed = get_seed()

    alpha_star = 0.5  # default 21cmfast value
    f_star10 = np.random.uniform(-1.45, -1.15)
    f_esc10 = np.random.uniform(-1.17, -0.84)
    alpha_esc = -0.5  # default 21cmfast value
    l_x = np.random.uniform(40.45, 40.55)

    initial_conditions = p21c.initial_conditions(
        user_params=user_params,
        cosmo_params=cosmoparams,
        random_seed=random_seed,
        direc="_cache",
        write=False,
    )

    lightcone = p21c.run_lightcone(
        redshift=zmin,
        max_redshift=zmax,
        init_box=initial_conditions,
        flag_options=flag_options,
        astro_params=p21c.AstroParams(
            {
                "F_STAR10": f_star10,
                "F_ESC10": f_esc10,
                "L_X": l_x,
            }
        ),
        lightcone_quantities=["brightness_temp"],
        direc="_cache",
        write=False,
    )

    T21 = lightcone.brightness_temp

    zs = lightcone.lightcone_redshifts[:HII_DIM]
    T21 = chop_box(T21)

    return T21, alpha_star, f_star10, f_esc10, alpha_esc, l_x, zs[0], zs[-1]


directory_name = "final_data_T21s_LC_z={:.1f}_{:d}Mpc_{:d}Cells_{:d}boxes".format(
    redshift, BOX_LEN, HII_DIM, N_cubes
)


def save_lightcones(N, filename="data/T21s_LC_z=10"):
    unique_id = uuid.uuid4()
    filename = f"{filename}_{unique_id}.npy"
    boxes = []
    for _ in range(N):
        T21, alpha_star, f_star10, f_esc10, alpha_esc, l_x, z_min, z_max = (
            generate_lightcone()
        )
        boxes.append(T21)
        save_parameters_to_txt(
            [alpha_star, f_star10, f_esc10, alpha_esc, l_x, z_min, z_max],
            filename=directory_name + "/parameters.txt",
        )
    boxes = np.array(boxes)
    np.save(filename, boxes)


# Check if the directory exists and create it if it does not
if not os.path.isdir(directory_name):
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except OSError as error:
        print(f"Creation of the directory '{directory_name}' failed due to: {error}")
else:
    print(f"Directory '{directory_name}' already exists, continuing...")

save_lightcones(
    N=N_cubes,
    filename=directory_name + "/run",
)

# use the line below if you want to clear cache after every run
p21c.cache_tools.clear_cache(direc="_cache")
