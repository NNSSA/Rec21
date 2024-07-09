import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
from unet import UNet3D
import matplotlib.pyplot as plt
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################################
######################## LOAD DATA ########################
###########################################################

T21s_wr = np.load(
    "/scratch4/mkamion1/nsabti1/21cm_ML/data/T21s_wr_z=15.0_128Mpc_32Cells_1000boxes_norelvel_psi=0.9_usingaugmentedT21.npy"
)[:10000]
T21s = np.load("/scratch4/mkamion1/nsabti1/21cm_ML/data/T21s_z=15.0_128Mpc_32Cells_1000boxes_norelvel_psi=0.9_augmented_2.npy")[:10000]

print(T21s_wr.shape)
print(T21s.shape)

psi = 50. / 180. * np.pi  # variable that defines the wedge
Nboxes = len(T21s_wr)
d = 32
Ntrain = 9900
Ntest = 100
batch_size = 5

def separate_real_imag(data, Nboxes, d):
    data = np.reshape(data, (Nboxes, 1, d, d, d))
    data_realimag = []
    for data_element in data:
        data_element_fft = np.fft.fftn(data_element)
        data_realimag.append(np.vstack((np.real(data_element_fft), np.imag(data_element_fft))))
    return np.array(data_realimag)

T21s_wr_realimag = separate_real_imag(T21s_wr, Nboxes, d)
T21s_realimag = separate_real_imag(T21s, Nboxes, d)

val_min = np.min(T21s_realimag)
val_max = np.max(T21s_realimag)
T21s_wr_realimag = (T21s_wr_realimag - val_min) / (val_max - val_min)
T21s_realimag = (T21s_realimag - val_min) / (val_max - val_min)
print(np.min(T21s_wr_realimag), np.max(T21s_wr_realimag), np.min(T21s_realimag), np.max(T21s_realimag))

def unnormalize(input_array):
    return input_array * (val_max - val_min) + val_min

X = torch.tensor(
    T21s_wr_realimag[:Ntrain],
    dtype=torch.float32,
    device=device,
)
Y = torch.tensor(
    T21s_realimag[:Ntrain], dtype=torch.float32, device=device
)

X_test = torch.tensor(
    T21s_wr_realimag[Ntrain:Ntrain+Ntest],
    dtype=torch.float32,
    device=device,
)
Y_test = torch.tensor(
    T21s_realimag[Ntrain:Ntrain+Ntest], dtype=torch.float32, device=device
)

print(X.shape, Y.shape, X_test.shape, Y_test.shape)


#######################################################
######################## MASKS ########################
#######################################################

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


def get_indices(arr_shape):
    original_indices = []
    conjugate_indices = []
    for dim_size in arr_shape:
        indices = np.arange(dim_size)
        indices[1:] = indices[1:][::-1]
        original_indices.append(np.arange(dim_size)) 
        conjugate_indices.append(indices)
    return original_indices, conjugate_indices

def create_conjugate_mask(input_shape):

    all_indices = get_indices(input_shape)

    pairs_of_indices = list(itertools.product(*all_indices[0]))
    pairs_of_indices_conjugate = list(itertools.product(*all_indices[1]))

    to_zero_out = set()
    for num in range(len(pairs_of_indices)):
        if pairs_of_indices_conjugate[num] in pairs_of_indices[:num]:
            to_zero_out.add(pairs_of_indices[num])

    mask = np.ones(input_shape, dtype=int)

    for idx in to_zero_out:
        mask[idx] = 0
        
    return mask

def reconstruct_original_fourier(input_array, mask):
    input_array = np.array(input_array[0] + input_array[1] * 1j)
    all_indices = get_indices(input_array.shape)
    return input_array + np.conj(input_array[np.ix_(*all_indices[1])]) * (1-mask)

def reconstruct_original_image(input_array, mask):
    input_array = np.array(input_array[0] + input_array[1] * 1j)
    all_indices = get_indices(input_array.shape)
    input_reconstructed = input_array + np.conj(input_array[np.ix_(*all_indices[1])]) * (1-mask)
    return np.real(np.fft.ifftn(input_reconstructed))

wedge_mask = torch.tensor(create_wedge_mask_fft((d,d,d))).to(device)
conjugate_mask = torch.tensor(create_conjugate_mask((d,d,d))).to(device)

def apply_masks_train(inputs, model_pred, actual_result):
    model_pred = inputs[:,:] * wedge_mask + model_pred[:,:] * (1 - wedge_mask)
    model_pred = model_pred[:,:] * conjugate_mask
    actual_result_masked = actual_result[:,:] * conjugate_mask
    return model_pred, actual_result_masked

def apply_masks_eval(inputs, model_pred):
    model_pred = inputs[:,:] * wedge_mask + model_pred[:,:] * (1 - wedge_mask)
    model_pred = model_pred[:,:] * conjugate_mask
    return model_pred

#######################################################
###################### NETWORK ########################
#######################################################

class Kernel:
    def __init__(self, Model) -> None:
        self.Model = Model
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.loss_fn = torch.nn.MSELoss()
        self.split_train_val(batch_size=batch_size)
        print(self.loss_fn(unnormalize(X),unnormalize(Y)).item())

    def split_train_val(self, batch_size) -> None:
        self.train_loader = torch.utils.data.DataLoader(
            [[X[i], Y[i]] for i in range(Ntrain)], batch_size=batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            [[X_test[i], Y_test[i]] for i in range(Ntest)], batch_size=Ntest, shuffle=True
        )

    def test_model(self, loss_test_list):
        self.Model.train(False)
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, actual_result = data

                model_pred = self.Model(inputs)
                model_pred_masked, actual_result_masked = apply_masks_train(inputs, model_pred, actual_result)
                loss = self.loss_fn(unnormalize(model_pred_masked), unnormalize(actual_result_masked))

                test_loss += loss.item()
            print("LOSS test {}".format(test_loss / (i + 1)))
            loss_test_list.append(test_loss / (i + 1))


    def train_model(self, optimizer, loss_train_list, loss_test_list, epochs):
        self.Model.train(True)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                inputs, actual_result = data[0].to(device), data[1].to(device)

                model_pred = self.Model(inputs)
                model_pred_masked, actual_result_masked = apply_masks_train(inputs, model_pred, actual_result)

                loss = self.loss_fn(unnormalize(model_pred_masked), unnormalize(actual_result_masked))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print("EPOCH {}:".format(epoch + 1))
            print("LOSS train {}".format(running_loss / (i + 1)))
            loss_train_list.append(running_loss / (i + 1))
            self.test_model(loss_test_list)

            if (epoch+1) % 20 == 0:
                optimizer.param_groups[0]['lr'] /= 2.
                print(optimizer.param_groups[0]['lr'])


Model = UNet3D(
    in_channels=2,
    out_classes=2,
    num_encoding_blocks=4,
    out_channels_first_layer=24,
    # dropout=0.4,
    normalization='instance',
    # upsampling_type='trilinear',
    preactivation=True,
    # residual=True,
    padding=1,
).to(device)
print("Total number of model params = %d" % sum(p.numel() for p in Model.parameters()))

# Model = torch.load(
#    "/scratch4/mkamion1/nsabti1/21cm_ML/models/Model_test.pt"
# )

#######################################################
###################### Training #######################
#######################################################

Train_kernel = Kernel(Model=Model)
loss_train_list = []
loss_test_list = []

lr = 1e-3
epochs = 15
optimizer = torch.optim.Adam(Model.parameters(), lr=lr)#, weight_decay=1e-8)
Train_kernel.train_model(optimizer, loss_train_list, loss_test_list, epochs)

torch.save(
    Model,
    "/scratch4/mkamion1/nsabti1/21cm_ML/models/Model_T21Fourier_z15psi0p9.pt",
)

# torch.save(
#     Model.state_dict(),
#     "/scratch4/mkamion1/nsabti1/21cm_ML/models/Model_test_dict.pt",
# )


plt.figure()
plt.semilogy(loss_train_list, color="blue", label="train loss")
plt.semilogy(loss_test_list, color="red", label="test loss")
plt.legend()
plt.savefig("loss_T21Fourier_z15psi0p9.png")

plt.figure()

#######################################################
#################### Evaluation #######################
#######################################################

T21s_test_pred = []
T21s_test_input = []
T21s_test_real = []
for i in range(Ntest):
    test_pred = Model(X_test[i:i+1])
    test_pred = apply_masks_eval(unnormalize(X_test[i:i+1]), unnormalize(test_pred)).cpu().detach().numpy().squeeze()
    test_pred_image = reconstruct_original_image(test_pred, conjugate_mask.cpu().detach().numpy())

    test_input = apply_masks_eval(unnormalize(X_test[i:i+1]), unnormalize(X_test[i:i+1])).cpu().detach().numpy().squeeze()
    test_input_image = reconstruct_original_image(test_input, conjugate_mask.cpu().detach().numpy())

    test_real = apply_masks_eval(unnormalize(Y_test[i:i+1]), unnormalize(Y_test[i:i+1])).cpu().detach().numpy().squeeze()
    test_real_image = reconstruct_original_image(test_real, conjugate_mask.cpu().detach().numpy())

    T21s_test_pred.append(test_pred_image)
    T21s_test_input.append(test_input_image)
    T21s_test_real.append(test_real_image)

T21s_test_pred = np.array(T21s_test_pred)
T21s_test_input = np.array(T21s_test_input)
T21s_test_real = np.array(T21s_test_real)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

title_font = {"family": "serif", "color": "black", "size": 16}
label_font = {"family": "serif", "color": "black", "weight": "normal", "size": 14}

for col, title in enumerate(["T21s_wr", "T21s", "T21s_pred"]):
    axes[0, col].set_title(title, fontdict=title_font)

for j in range(2):
    q = np.random.randint(0, Ntest)
    vmax = np.max(T21s_test_real[q])
    vmin = np.min(T21s_test_real[q])
    axes[j][0].imshow(T21s_test_input[q][0,:,:])
    axes[j][1].imshow(T21s_test_real[q][0,:,:])
    axes[j][2].imshow(T21s_test_pred[q][0,:,:])
    print("MSE for j = {} is {}".format(j, ((T21s_test_pred[q] - T21s_test_real[q])**2).mean()))

for j in range(2):
    for i in range(3):
        axes[j, i].axis("off")


fig.suptitle("Slices along LoS", fontsize=18, y=1.03)
plt.tight_layout()

plt.savefig("plot_T21Fourier_z15psi0p9.png")

