import torch
import torch.nn as nn
import numpy as np
from CustomUNET import UNet3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


divider = 255.0

# T21s_wr = np.load(
#     "/scratch4/mkamion1/nsabti1/21cm_ML/data/T21s_wr_z=15.0_128Mpc_32Cells_24000boxes_norelvel_psi=65.0_augmented.npy"
# ) / divider
T21s = np.load(
    "/scratch4/mkamion1/nsabti1/21cm_ML/data/T21s_z=15.0_128Mpc_32Cells_1000boxes_norelvel_psi=65.0.npy"
)[
    :8
]  # / divider

# print(T21s_wr.shape)
print(T21s.shape)

d = 32

Ntrain = 8
# Ntest = 100

# X = torch.tensor(
#     np.reshape(T21s_wr[:Ntrain], (Ntrain, 1, d, d, d)),
#     dtype=torch.float32,
#     device=device,
# )
Y = torch.tensor(
    np.reshape(T21s[:Ntrain], (Ntrain, 1, d, d, d)), dtype=torch.float32, device=device
)

# X_test = torch.tensor(
#     np.reshape(T21s_wr[Ntrain:Ntrain+Ntest], (Ntest, 1, d, d, d)),
#     dtype=torch.float32,
#     device=device,
# )
# Y_test = torch.tensor(
#     np.reshape(T21s[Ntrain:Ntrain+Ntest], (Ntest, 1, d, d, d)), dtype=torch.float32, device=device
# )


###################################################
###################################################
################# Define Network ##################
###################################################
###################################################


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

    def forward(
        self,
        time,
        x,
    ):
        # Get gamma to shape (B, ).
        time = time.expand(x.shape[0])  # assume shape () or (1,) or (B,)
        assert time.shape == (x.shape[0],)
        # Rescale to [0, 1], but only approximately since gamma0 & gamma1 are not fixed.
        time = (time - torch.min(time)) / (torch.max(time) - torch.min(time))
        t_embedding = get_timestep_embedding(time, self.embedding_dim)
        # We will condition on time embedding.
        cond = self.embed_conditioning(t_embedding).to(device)
        x = self.conv_in(x)  # (B, embedding_dim, H, W)
        x = self.UNET(x, cond)
        return x


MyModel(
    in_channels=1,
    out_channels=1,
    num_blocks=2,
    out_channels_first_layer=4,
    embedding_dim=48,
    cond_embed_pref=4,
    dropout=0.4,
    normalization="instance",
    preactivation=True,
    padding=1,
).to(device)

MM = MyModel()
time = torch.rand(Y.shape[0]).type_as(Y)
MM(time, Y)
