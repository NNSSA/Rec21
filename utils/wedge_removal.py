import numpy as np

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


def create_wedge_mask_fft(input_shape):
    # For N_x = N_y = N_z

    N = input_shape[0]
    freq = np.fft.fftfreq(N)

    mask = np.ones(input_shape, dtype=int)
    indices_to_zero_out = set()

    def rule_remove_wedge(i, j, k):
        if np.abs(freq[i]) < np.sqrt(freq[j]**2 + freq[k]**2) * np.tan(psi):
            indices_to_zero_out.add((i,j,k))

    rule_vectorized = np.vectorize(rule_remove_wedge)
    indices = [
        np.arange(input_shape[0])[:, np.newaxis, np.newaxis],
        np.arange(input_shape[1])[:, np.newaxis],
        np.arange(input_shape[2]),
    ]
    rule_vectorized(*indices)
    for idx in indices_to_zero_out:
        mask[idx] = 0 

    return mask
