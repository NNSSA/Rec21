import torch
import numpy as np


class SDEIntegrator:
    def __init__(self, drift_model, t_span, n_step, n_save, device):
        self.drift_model = drift_model
        self.t_span = t_span
        self.dt = self.t_span[1] - self.t_span[0]
        self.n_step = n_step
        self.n_save = n_save
        self.device = device

    def step_forward_heun(self, t, x):
        dW = torch.sqrt(self.dt) * torch.randn(
            size=x.shape, device=self.device
        )  # .to(device)
        xhat = x + (1.0 - t) * dW
        K1 = self.drift_model(t + self.dt, xhat)
        xp = xhat + self.dt * K1
        K2 = self.drift_model(t + self.dt, xp)
        return xhat + 0.5 * self.dt * (K1 + K2)

    def step_forward(self, t, x):
        dW = torch.sqrt(self.dt) * torch.randn(
            size=x.shape, device=self.device
        )  # .to(device)
        return x + self.drift_model(t, x) * self.dt + (1.0 - t) * dW

    def rollout_forward(self, x0, method="heun"):
        save_every = int(self.n_step / self.n_save)
        xs = torch.zeros((self.n_save, *x0.shape), device=self.device)  # .to(device)
        x = x0.to(self.device)

        save_counter = 0

        for ii, t in enumerate(self.t_span[:-1]):
            if method == "heun":
                x = self.step_forward_heun(t, x)
            else:
                x = self.step_forward(t, x)

            if ((ii + 1) % save_every) == 0:
                xs[save_counter] = x
                save_counter += 1

        xs[save_counter] = x

        return xs


def adjust_learning_rate(optimizer, epoch, args):
    lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))
