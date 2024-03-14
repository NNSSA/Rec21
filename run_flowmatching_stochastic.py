import torch
import torch.nn as nn
import numpy as np
from torchdyn.core import NeuralODE
from torchdiffeq import odeint, odeint_adjoint
import torchsde
from CustomUNET import UNet3D
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################
################# Data ##################
#########################################

divider = 1.0  # 255.0

T21s_wr = (
    np.load(
        "/scratch4/mkamion1/nsabti1/21cm_ML/data/T21s_wr_z=15.0_128Mpc_32Cells_24000boxes_norelvel_psi=65.0_augmented.npy"
    )
    / divider
)
T21s = (
    np.load(
        "/scratch4/mkamion1/nsabti1/21cm_ML/data/T21s_z=15.0_128Mpc_32Cells_24000boxes_norelvel_psi=65.0_augmented.npy"
    )
    / divider
)

# means = np.mean(T21s_wr)
# stds = np.std(T21s_wr)
# T21s_wr = (T21s_wr - means) / stds
# T21s = (T21s - means) / stds

print(T21s_wr.shape)
print(T21s.shape)

d = 32

Ntrain = 23000
Ntest = 1000
Nbatch = 10

T21s_wr_train = torch.tensor(
    np.reshape(T21s_wr[:Ntrain], (Ntrain, 1, d, d, d)),
    dtype=torch.float32,
    device=device,
)
T21s_train = torch.tensor(
    np.reshape(T21s[:Ntrain], (Ntrain, 1, d, d, d)), dtype=torch.float32, device=device
)

T21s_wr_test = torch.tensor(
    np.reshape(T21s_wr[Ntrain : Ntrain + Ntest], (Ntest, 1, d, d, d)),
    dtype=torch.float32,
    device=device,
)
T21s_test = torch.tensor(
    np.reshape(T21s[Ntrain : Ntrain + Ntest], (Ntest, 1, d, d, d)),
    dtype=torch.float32,
    device=device,
)

train_mean = torch.mean(torch.cat((T21s_wr_train, T21s_train)))
train_std = torch.std(torch.cat((T21s_wr_train, T21s_train)))
T21s_wr_train = (T21s_wr_train - train_mean) / train_std
T21s_train = (T21s_train - train_mean) / train_std
T21s_wr_test = (T21s_wr_test - train_mean) / train_std
T21s_test = (T21s_test - train_mean) / train_std

train_loader = torch.utils.data.DataLoader(
    [[T21s_wr_train[i], T21s_train[i]] for i in range(Ntrain)],
    batch_size=Nbatch,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    [[T21s_wr_test[i], T21s_test[i]] for i in range(Ntest)],
    batch_size=Nbatch,
    shuffle=True,
)


###################################################
################# Define Network ##################
###################################################


@torch.no_grad()
def zero_init(module: torch.nn.Module) -> torch.nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        torch.nn.init.zeros_(p.data)
    # return module


def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10000,
    min_timescale=1,
):
    # Scale timesteps by a factor of 1000
    timesteps *= 1000
    # Ensure timesteps is a 1-dimensional tensor
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0

    num_timescales = embedding_dim // 2
    # Create a tensor of inverse timescales logarithmically spaced
    inv_timescales = torch.logspace(
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # Shape: (T, D/2)
    # Return the concatenation of sine and cosine of the embedding
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # Shape: (T, D)


class MyModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_blocks: int = 2,
        out_channels_first_layer=4,
        embedding_dim: int = 48,
        cond_embed_pref: int = 4,
        dropout=0.4,
        normalization="instance",
        preactivation=True,
        padding=1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embed_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * cond_embed_pref),
            nn.SiLU(),
            nn.Linear(embedding_dim * cond_embed_pref, embedding_dim * cond_embed_pref),
            nn.SiLU(),
        ).to(device)

        self.conv_in = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=3,
            padding=1,
        ).to(device)
        self.UNET = UNet3D(
            in_channels=embedding_dim,
            out_classes=out_channels,
            num_encoding_blocks=num_blocks,
            out_channels_first_layer=out_channels_first_layer,
            condition_dim=embedding_dim * cond_embed_pref,
            dropout=dropout,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
        ).to(device)
        # zero_init(list(list(self.UNET.children())[-1:][0].children())[0])

    def forward(
        self,
        time,
        x,
    ):
        # Get gamma to shape (B, ).
        time = time.expand(x.shape[0])  # assume shape () or (1,) or (B,)
        assert time.shape == (x.shape[0],)
        # Rescale to [0, 1], but only approximately since gamma0 & gamma1 are not fixed.
        # time = (time - self.time_min) / (self.time_max - self.time_min)
        t_embedding = get_timestep_embedding(time, self.embedding_dim)
        # We will condition on time embedding.
        cond = self.embed_conditioning(t_embedding).to(device)
        x = self.conv_in(x)  # (B, embedding_dim, H, W)
        x = self.UNET(x, cond)
        return x


##################################################
################# Flow Matching ##################
##################################################


# class torch_wrapper(torch.nn.Module):
#     """Wraps model to torchdyn compatible format."""

#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, t, x, *args, **kwargs):
#         return self.model(t, x)


class SDEIntegrator:
    def __init__(self, drift_model, t_span, n_step, n_save):
        """Initialize forward dynamics, reverse dynamics, and likelihood."""

        self.drift_model = drift_model
        self.t_span = t_span
        self.dt = self.t_span[1] - self.t_span[0]
        self.n_step = n_step
        self.n_save = n_save

    def step_forward_heun(self, t, x):
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW = torch.sqrt(self.dt) * torch.randn(size=x.shape).to(device)
        xhat = x + (1.0 - t) * dW
        K1 = self.drift_model(t + self.dt, xhat)
        xp = xhat + self.dt * K1
        K2 = self.drift_model(t + self.dt, xp)
        return xhat + 0.5 * self.dt * (K1 + K2)

    def step_forward(self, t, x):
        """Euler-Maruyama."""
        dW = torch.sqrt(self.dt) * torch.randn(size=x.shape).to(device)
        return x + self.drift_model(t, x) * self.dt + (1.0 - t) * dW

    def rollout_forward(self, x0, method="heun"):
        save_every = int(self.n_step / self.n_save)
        xs = torch.zeros((self.n_save, *x0.shape)).to(device)
        x = x0
        # self.dt = self.dt.to(x)

        save_counter = 0

        for ii, t in enumerate(self.t_span[:-1]):
            # t = t.to(x)
            # t = t.unsqueeze(0)
            if method == "heun":
                x = self.step_forward_heun(t, x)
            else:
                x = self.step_forward(t, x)

            if ((ii + 1) % save_every) == 0:
                xs[save_counter] = x
                save_counter += 1

        xs[save_counter] = x

        return xs


# class SDE(nn.Module):

#     def __init__(self, drift):
#         super().__init__()
#         self.drift = drift
#         self.noise_type = "diagonal"
#         self.sde_type = "ito"

#     def f(self, t, x):
#         return self.drift(t, x)

#     def g(self, t, x):
#         return 1. - t


class FlowMatching(torch.nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model
        self.loss_test_list = []
        self.loss_train_list = []
        # self.loss_fn = torch.nn.MSELoss()

    def loss_fn(self, model, true):
        return torch.mean(torch.abs(model - true) ** 2)

    def get_x_t(self, x0, x1, eps, t):
        # t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        return t**2 * x1 + (1 - t) * x0 + torch.sqrt(t) * (1 - t) * eps

    def compute_loss(
        self,
        x0,
        x1,
        t=None,
    ):
        if t is None:
            t = torch.distributions.Uniform(0.0, 1.0).sample((x0.shape[0],)).type_as(x0)
        t2 = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        eps = torch.randn_like(x0)
        xt = self.get_x_t(x0, x1, eps, t2)
        bt = 2.0 * t2 * x1 - x0 - torch.sqrt(t2) * eps
        vt = self.flow_model(t, xt)
        return self.loss_fn(vt, bt)

    def test_model(self):
        self.flow_model.train(False)
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, actual_result = data[0].to(device), data[1].to(device)
                # inputs, actual_result = data

                loss = self.compute_loss(inputs, actual_result)
                test_loss += loss.item() * divider**2

            print("LOSS test {}".format(test_loss / (i + 1)))
            self.loss_test_list.append(test_loss / (i + 1))

    def train_model(self, optimizer, epochs):
        self.flow_model.train(True)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, actual_result = data[0].to(device), data[1].to(device)

                loss = self.compute_loss(inputs, actual_result)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item() * divider**2

            print("EPOCH {}:".format(epoch + 1))
            print("LOSS train {}".format(running_loss / (i + 1)))
            self.loss_train_list.append(running_loss / (i + 1))
            self.test_model()

            if (epoch + 1) % 5 == 0:
                optimizer.param_groups[0]["lr"] /= 2.0
                print(optimizer.param_groups[0]["lr"])

    def sample(self, x0, n_steps=100, n_save=5, method="heun"):
        t_span = torch.linspace(0.0, 1.0, n_steps).to(device)
        sde = SDEIntegrator(self.flow_model, t_span, n_steps, n_save)
        with torch.no_grad():
            traj = sde.rollout_forward(x0, method)
        return traj


flow_model = MyModel(
    in_channels=1,
    out_channels=1,
    num_blocks=4,
    out_channels_first_layer=24,
    embedding_dim=48,
    cond_embed_pref=4,
    dropout=0.3,
    normalization="instance",
    preactivation=True,
    padding=1,
).to(device)
print(
    "Total number of model params = %d"
    % sum(p.numel() for p in flow_model.parameters())
)


################################################
################# Train Model ##################
################################################


FM_kernel = FlowMatching(flow_model=flow_model)
lr = 1e-3
epochs = 10
optimizer = torch.optim.Adam(flow_model.parameters(), lr=lr)  # , weight_decay=1e-8)
print("Identity loss: ", FM_kernel.loss_fn(T21s_wr_train, T21s_train))
FM_kernel.train_model(optimizer, epochs)

torch.save(
    flow_model,
    "/scratch4/mkamion1/nsabti1/21cm_ML/models/Flowmodel_T21_z15psi65_SDE_1.pt",
)

# flow_model = torch.load(
#    "/scratch4/mkamion1/nsabti1/21cm_ML/models/Flowmodel_T21_z15psi65_SDE_1.pt"
# )
# FM_kernel = FlowMatching(flow_model=flow_model)


#############################################
################# Plotting ##################
#############################################


plt.figure()
plt.semilogy(FM_kernel.loss_train_list, color="blue", label="train loss")
plt.semilogy(FM_kernel.loss_test_list, color="red", label="test loss")
plt.legend()
# plt.axis(ymax=20)
plt.savefig("loss_flowmatching_z15psi65_SDE_1.png")

x0s, x1s = next(iter(test_loader))
x0s.to(device)
x1s.to(device)

n_steps = 500
n_save = 5
t_span = torch.linspace(0.0, 1.0, n_steps)
fig, ax = plt.subplots(nrows=3, ncols=n_save + 2, figsize=(20, 6))
numrows = 3
randnums = np.random.choice(
    Nbatch - 1, numrows, replace=False
)  # randint(0, Nbatch - 1)

for row in range(numrows):
    x0_to_use = x0s[randnums[row]][None]
    x1_to_use = x1s[randnums[row]][None]
    sampled_images = FM_kernel.sample(x0_to_use, n_steps=n_steps, n_save=n_save)

    # Initial image
    ax[row, 0].set_title(f"T21_wr")
    ax[row, 0].imshow(x0_to_use.detach().cpu().numpy()[0][0][10, :, :].squeeze())
    ax[row, 0].set_xticks([])
    ax[row, 0].set_yticks([])

    # Fill the middle plots with sampled images
    for i in range(n_save):
        step = int(n_steps / n_save) * (i + 1)
        ax[row, i + 1].set_title(f"Steps = {step}")
        ax[row, i + 1].imshow(
            sampled_images[i].detach().cpu().numpy()[0][0][10, :, :].squeeze()
        )
        ax[row, i + 1].set_xticks([])
        ax[row, i + 1].set_yticks([])

    # Final image
    ax[row, -1].set_title(f"T21")
    ax[row, -1].imshow(x1_to_use.detach().cpu().numpy()[0][0][10, :, :].squeeze())
    ax[row, -1].set_xticks([])
    ax[row, -1].set_yticks([])

plt.savefig("reconstruction_flowmatching_z15psi65_SDE_1.png")
