import torch
from types import SimpleNamespace
from exp import Exp_Reconstruct
import random
import numpy as np

# fix seed for reproducibility's sake
fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args = SimpleNamespace(
    model_id="Rec21_200k_finalfinal_5epochs_halfdata_48M_samples",
    # data loader
    file_path="data_highres/final_data_LC_z=10.0_512Mpc_128Cells_200120boxes.h5",
    T21_name="T21_data",
    T21_wr_name="T21_wr_data",
    train_ratio=0.9,
    # training
    train_from_scratch=False,
    id_pretained_model="Rec21_200k_finalfinal_5epochs_halfdata",
    # sampling (only if train_from_scratch = False)
    write_samples=True,  # set to False if want to plot a few samples
    N_samples=1000,
    # model parameters
    in_channels=1,
    out_channels=1,
    num_blocks=4,
    out_channels_first_layer=48,
    embedding_dim=48,
    cond_embed_pref=4,
    dropout=0.0,
    normalization="instance",
    preactivation=True,
    padding=1,
    # optimization
    learning_rate=0.001,
    batch_size=3,
    train_epochs=10,  # 5,
    num_workers=4,
    # GPU
    use_gpu=True,
    gpu=0,
    use_multi_gpu=False,
    devices="0,1,2,3",
)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(" ", "")
    device_ids = args.devices.split(",")
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# define training/testing/predicting class
Exp = Exp_Reconstruct
exp = Exp(args)

# training bit
print(">>>>>>> Training arguments >>>>>>>")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("\n")

if args.train_from_scratch:
    exp.train_model()
else:
    if args.write_samples:
        exp.sample_N_boxes(args.N_samples)
    else:
        exp.plot_sample()

print(">>>>>>> Finished, results saved in output folder >>>>>>>")

torch.cuda.empty_cache()
