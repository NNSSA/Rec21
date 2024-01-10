import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
from unet import UNet3D
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

divider = 255.

T21s_wr = np.load(
    "/scratch4/mkamion1/nsabti1/21cm_ML/data/T21s_wr_z=15.0_128Mpc_32Cells_1000boxes_norelvel_psi=0.9_usingaugmentedT21.npy"
) / divider
T21s = np.load("/scratch4/mkamion1/nsabti1/21cm_ML/data/T21s_z=15.0_128Mpc_32Cells_1000boxes_norelvel_psi=0.9_augmented_2.npy") / divider

print(T21s_wr.shape)
print(T21s.shape)

d = 32

Ntrain = 23900
Ntest = 100

X = torch.tensor(
    np.reshape(T21s_wr[:Ntrain], (Ntrain, 1, d, d, d)),
    dtype=torch.float32,
    device=device,
)
Y = torch.tensor(
    np.reshape(T21s[:Ntrain], (Ntrain, 1, d, d, d)), dtype=torch.float32, device=device
)

X_test = torch.tensor(
    np.reshape(T21s_wr[Ntrain:Ntrain+Ntest], (Ntest, 1, d, d, d)),
    dtype=torch.float32,
    device=device,
)
Y_test = torch.tensor(
    np.reshape(T21s[Ntrain:Ntrain+Ntest], (Ntest, 1, d, d, d)), dtype=torch.float32, device=device
)

class Kernel:
    def __init__(self, Model) -> None:
        self.Model = Model
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.loss_fn = torch.nn.MSELoss()
        self.split_train_val(batch_size=5)
        print(self.loss_fn(X,Y).item() * divider**2)

    # def dice_coef(self, y_pred, y_true):
    #     y_true_f = y_true.flatten()
    #     y_pred_f = y_pred.flatten()
    #     intersection = torch.sum(y_true_f * y_pred_f)
    #     smooth = 1.
    #     return 1.-(2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

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
                loss = self.loss_fn(model_pred, actual_result)

                test_loss += loss.item() * divider**2
            print("LOSS test {}".format(test_loss / (i + 1)))
            loss_test_list.append(test_loss / (i + 1))


    def train_model(self, optimizer, loss_train_list, loss_test_list, epochs):
        self.Model.train(True)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                inputs, actual_result = data[0].to(device), data[1].to(device)

                model_pred = self.Model(inputs)
                loss = self.loss_fn(model_pred, actual_result)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * divider**2

            print("EPOCH {}:".format(epoch + 1))
            print("LOSS train {}".format(running_loss / (i + 1)))
            loss_train_list.append(running_loss / (i + 1))
            self.test_model(loss_test_list)

            if (epoch+1) % 5 == 0:
                optimizer.param_groups[0]['lr'] /= 2.
                print(optimizer.param_groups[0]['lr'])


Model = UNet3D(
    in_channels=1,
    out_classes=1,
    num_encoding_blocks=4,
    out_channels_first_layer=24,
    dropout=0.4,
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


Train_kernel = Kernel(Model=Model)
loss_train_list = []
loss_test_list = []

lr = 1e-3
epochs = 10
optimizer = torch.optim.Adam(Model.parameters(), lr=lr, weight_decay=1e-8)
Train_kernel.train_model(optimizer, loss_train_list, loss_test_list, epochs)

torch.save(
    Model,
    "/scratch4/mkamion1/nsabti1/21cm_ML/models/Model_T21augmented_z15psi0p9_usingaugmentedT21.pt",
)

# torch.save(
#     Model.state_dict(),
#     "/scratch4/mkamion1/nsabti1/21cm_ML/models/Model_test_dict.pt",
# )


plt.figure()
plt.semilogy(loss_train_list, color="blue", label="train loss")
plt.semilogy(loss_test_list, color="red", label="test loss")
plt.legend()
plt.axis(ymax=20)
plt.savefig("loss_T21augmented_z15psi0p_usingaugmentedT21.png")

plt.figure()

################################################
################################################


T21s_pred = []
for i in range(Ntest):
    x = np.reshape(Model(X_test[i:i+1]).cpu().detach().numpy(), (d, d, d))
    T21s_pred.append(x)

T21s_pred = np.array(T21s_pred)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

title_font = {"family": "serif", "color": "black", "size": 16}
label_font = {"family": "serif", "color": "black", "weight": "normal", "size": 14}

for col, title in enumerate(["T21s_wr", "T21s", "T21s_pred"]):
    axes[0, col].set_title(title, fontdict=title_font)

for j in range(2):
    q = np.random.randint(0, Ntest)
    vmax = np.max(Y_test[q][0].cpu().detach().numpy())
    vmin = np.min(Y_test[q][0].cpu().detach().numpy())
    axes[j][0].imshow(X_test[q][0][0,:,:].cpu().detach().numpy())
    axes[j][1].imshow(Y_test[q][0][0,:,:].cpu().detach().numpy())
    axes[j][2].imshow(T21s_pred[q][0,:,:])

for j in range(2):
    for i in range(3):
        axes[j, i].axis("off")


fig.suptitle("Slices along LoS", fontsize=18, y=1.03)
plt.tight_layout()

plt.savefig("plot_T21augmented_z15psi0p9_usingaugmentedT21.png")

