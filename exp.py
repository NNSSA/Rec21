import torch
import torch.nn as nn
from torch import optim
from model import MyModel
from utils import SDEIntegrator, adjust_learning_rate
import data_loader
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class Exp_Reconstruct(object):
    def __init__(self, args):

        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total number of parameters in the model:", total_params, "\n")

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        model = MyModel(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if not self.args.train_from_scratch:
            model.load_state_dict(
                torch.load(
                    "./output/Trained_model_{}.pth".format(self.args.id_pretained_model)
                )
            )

        return model

    def _get_data(self):
        train_loader, test_loader = data_loader.data_provider(self.args)
        return train_loader, test_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _loss_fn(self, model, true):
        return torch.mean(torch.abs(model - true) ** 2)

    def get_x_t(self, x0, x1, eps, t):
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
        vt = self.model(t, xt)
        return self._loss_fn(vt, bt)

    def test_model(self, test_loader):
        test_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (T21_wr, T21_full) in enumerate(test_loader):
                T21_wr, T21_full = T21_wr.float().to(self.device), T21_full.float().to(
                    self.device
                )
                loss = self.compute_loss(T21_wr, T21_full)
                test_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\tbatch: {0} / {1}".format(i + 1, len(test_loader)))

        test_loss = np.average(test_loss)
        print("Test loss: ", test_loss)
        self.model.train(True)
        return test_loss

    def train_model(self):
        print(">>>>>>> Loading train and test data >>>>>>>")
        self.train_loader, self.test_loader = self._get_data()
        self.loss_test_list = []
        self.loss_train_list = []
        optimizer = self._select_optimizer()

        print("\n")
        print(">>>>>>> Start training >>>>>>>")
        self.model.train(True)
        for epoch in range(self.args.train_epochs):
            train_running_loss = 0.0
            for i, (T21_wr, T21_full) in enumerate(self.train_loader):
                T21_wr, T21_full = T21_wr.float().to(self.device), T21_full.float().to(
                    self.device
                )
                loss = self.compute_loss(T21_wr, T21_full)

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_running_loss += loss.item()

                if (i + 1) % 1000 == 0:
                    print(
                        "\tbatch: {0} / {1}, epoch: {2} | loss: {3:.7f}".format(
                            i + 1, len(self.train_loader), epoch + 1, loss.item()
                        )
                    )

            print("EPOCH {}:".format(epoch + 1))
            print("Train loss {}".format(train_running_loss / (i + 1)))
            self.loss_train_list.append(train_running_loss / (i + 1))
            print(">>>>>>> Start testing >>>>>>>")
            test_loss = self.test_model(self.test_loader)
            self.loss_test_list.append(test_loss)

            if (epoch + 1) % 3 == 0:
                optimizer.param_groups[0]["lr"] /= 1.5
                print(optimizer.param_groups[0]["lr"])

            torch.save(
                self.model.state_dict(),
                "./output/Trained_model_{}.pth".format(self.args.model_id),
            )

            # adjust_learning_rate(optimizer, epoch + 1, self.args)

        self.plot_loss()
        self.plot_sample()
        self.train_loader.dataset.close()
        self.test_loader.dataset.close()

    def sample(self, x0, n_steps=100, n_save=5, method="heun"):
        t_span = torch.linspace(0.0, 1.0, n_steps).to(self.device)
        sde = SDEIntegrator(self.model, t_span, n_steps, n_save, self.device)
        with torch.no_grad():
            traj = sde.rollout_forward(x0, method)
        return traj

    def plot_loss(self):
        plt.figure()
        plt.semilogy(self.loss_train_list, color="blue", label="train loss")
        plt.semilogy(self.loss_test_list, color="red", label="test loss")
        plt.legend()
        # plt.axis(ymax=20)
        plt.savefig("./output/loss_{}.png".format(self.args.model_id))

    def plot_sample(self):
        if not self.args.train_from_scratch:
            self.train_loader, self.test_loader = self._get_data()

        n_steps = 500
        n_save = 5
        num_rows = 3
        fig, ax = plt.subplots(nrows=num_rows, ncols=n_save + 2, figsize=(20, 6))

        for row in range(num_rows):
            x0_to_use, x1_to_use = next(iter(self.test_loader))
            x0_to_use, x1_to_use = x0_to_use[:1], x1_to_use[:1]

            sampled_images = self.sample(x0_to_use, n_steps=n_steps, n_save=n_save)

            # Initial image
            ax[row, 0].set_title(f"T21_wr")
            ax[row, 0].imshow(
                x0_to_use.detach().cpu().numpy()[0][0][10, :, :].squeeze()
            )
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
            ax[row, -1].imshow(
                x1_to_use.detach().cpu().numpy()[0][0][10, :, :].squeeze()
            )
            ax[row, -1].set_xticks([])
            ax[row, -1].set_yticks([])

        plt.savefig("./output/reconstruction_sample_{}.png".format(self.args.model_id))

    def sample_N_boxes(self, Nboxes):
        if not self.args.train_from_scratch:
            self.train_loader, self.test_loader = self._get_data()

        n_steps = 500
        n_save = 2
        x0_to_use, x1_to_use = next(iter(self.test_loader))
        x0_to_use, x1_to_use = x0_to_use[:1], x1_to_use[:1]

        dir_name = "./output/samples_{}".format(self.args.model_id)
        os.makedirs(dir_name, exist_ok=True)
        np.save(dir_name + "/x0.npy", x0_to_use.detach().cpu().numpy()[0][0].squeeze())
        np.save(dir_name + "/x1.npy", x1_to_use.detach().cpu().numpy()[0][0].squeeze())

        for i in range(Nboxes):
            sampled_images = self.sample(x0_to_use, n_steps=n_steps, n_save=n_save)
            final_image = sampled_images[-1].detach().cpu().numpy()[0][0].squeeze()

            # Save each sampled image
            np.save(dir_name + f"/sample_{i}.npy", final_image)
