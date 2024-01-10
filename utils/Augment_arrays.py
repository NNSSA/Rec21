import numpy as np

HII_DIM = 32
BOX_LEN = 128
N_cubes = 1000
redshift = 15.
psi = 50. / 180. * np.pi  # variable that defines the wedge

vcbs = np.load(
    "Data/vcbs_z={:.1f}_{:d}Mpc_{:d}Cells_{:d}boxes_norelvel_psi={:.1f}.npy".format(
        redshift, BOX_LEN, HII_DIM, N_cubes, psi
    )
)
T21s = np.load(
    "Data/T21s_z={:.1f}_{:d}Mpc_{:d}Cells_{:d}boxes_norelvel_psi={:.1f}.npy".format(
        redshift, BOX_LEN, HII_DIM, N_cubes, psi
    )
)

def rotations24(polycube):
    def rotations4(polycube, axes):
        for i in range(4):
             yield np.rot90(polycube, i, axes)

    yield from rotations4(polycube, (1,2))
    yield from rotations4(np.rot90(polycube, 2, axes=(0,2)), (1,2))
    yield from rotations4(np.rot90(polycube, axes=(0,2)), (0,1))
    yield from rotations4(np.rot90(polycube, -1, axes=(0,2)), (0,1))
    yield from rotations4(np.rot90(polycube, axes=(0,1)), (0,2))
    yield from rotations4(np.rot90(polycube, -1, axes=(0,1)), (0,2))


vcbs_augmented = []
T21s_augmented = []

for num in range(len(vcbs)):
    vcbs_augmented.extend(np.array(list(rotations24(vcbs[num]))))
    T21s_augmented.extend(np.array(list(rotations24(T21s[num]))))

np.save(
    "vcbs_z={:.1f}_{:d}Mpc_{:d}Cells_{:d}boxes_norelvel_psi={:.1f}_augmented.npy".format(
        redshift, BOX_LEN, HII_DIM, N_cubes, psi
    ),
    vcbs_augmented,
)
np.save(
    "T21s_z={:.1f}_{:d}Mpc_{:d}Cells_{:d}boxes_norelvel_psi={:.1f}_augmented.npy".format(
        redshift, BOX_LEN, HII_DIM, N_cubes, psi
    ),
    T21s_augmented,
)
