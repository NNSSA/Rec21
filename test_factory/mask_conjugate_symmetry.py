import numpy as np
import itertools

# function that return indices of original array and the indices required to create conjugate array
def get_indices(arr_shape):
    original_indices = []
    conjugate_indices = []
    for dim_size in arr_shape:
        indices = np.arange(dim_size)
        indices[1:] = indices[1:][::-1]
        original_indices.append(np.arange(dim_size)) 
        conjugate_indices.append(indices)
    return original_indices, conjugate_indices

def create_mask(input_shape):

    # get indices of original and its conjugate
    all_indices = get_indices(input_shape)

    # create all possible pairs of the original indices, i.e., (0,0), (0,1), ..., (1,0), (1,1),...
    pairs_of_indices = list(itertools.product(*all_indices[0]))
    # and similarly with the indices of the conjugate array
    pairs_of_indices_conjugate = list(itertools.product(*all_indices[1]))

    # idea here is to check which conjugate indices have already appeared in the original indices at position num. If True, the current original indices at num have already a conjugate and we can append the current position to be set equal to 0 later on.
    # for example: original indices = [(0,0), (0,1), (0,2), (0,3)] and conjugate indices = [(0,0), (0,3), (0,2), (0,1)]
    # as we iteratre over original indices, by the time we arrive at (0,3), the corresponding conjugate is (0,1), which has already appeared in original indices
    to_zero_out = set()
    for num in range(len(pairs_of_indices)):
        if pairs_of_indices_conjugate[num] in pairs_of_indices[:num]:
            to_zero_out.add(pairs_of_indices[num])

    # define mask
    mask = np.ones(input_shape, dtype=int)

    # set those elements of the mask = 0 where we already have a conjugate symmetric element
    for idx in to_zero_out:
        mask[idx] = 0
        
    return mask

# function to reconstruct the original array
def reconstruct_original(input_array, mask):
    all_indices = get_indices(input_array.shape)
    return input_array + np.conj(input_array[np.ix_(*all_indices[1])]) * (1-mask)

if __name__ == "__main__":
    input_array = np.random.random((32,32,32))#np.load("T21s_z=15.0_128Mpc_32Cells_1boxes_norelvel_psi=0.9.npy")[0]
    input_array_fft = np.fft.fftn(input_array)
    mask = create_mask(input_array_fft.shape)

    reconstructed_input_array_fft = reconstruct_original(input_array_fft * mask, mask)

    assert (np.isclose(input_array_fft, reconstructed_input_array_fft)).all(), "not working in Fourier space"
    assert (np.isclose(input_array, np.real(np.fft.ifftn(reconstructed_input_array_fft)))).all(), "not working in image space"

    print("mask = ", "\n", mask)
