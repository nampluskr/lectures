import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def gradient(y, x):
    """ return dy/dx """
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                create_graph=True, retain_graph=True)[0]


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh()):
        super().__init__()
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        if self.input_size > 1:
            inputs = torch.hstack(inputs)
        return self.model(inputs)


class Trainer:
    def __init__(self, model, optimizer, loss_functions={}, targets={}):
        self.model = model
        self.optimizer = optimizer
        self.loss_functions = loss_functions
        self.targets = targets
        self.mse = nn.MSELoss()
        # self.device = next(model.parameters()).device

        self.history = {"total": []}
        for name in self.loss_functions:
            self.history[name] = []
        for name in self.targets:
            self.history[name] = []

        self.loss_weights = {name: 1 for name in self.history}

    def fit(self, inputs, n_epochs, scheduler=None, update_step=10):
        with tqdm(range(1, n_epochs+1), file=sys.stdout, ascii=True, ncols=200) as pbar:
            for epoch in pbar:
                total_loss = 0

                for name in self.loss_functions:
                    loss_value = self.loss_functions[name](self.model, inputs)
                    loss_value *= self.loss_weights[name]
                    self.history[name].append(loss_value.item())
                    total_loss += loss_value

                for name in self.targets:
                    target_inputs, target_output = self.targets[name]
                    loss_target = self.mse(self.model(target_inputs), target_output)
                    loss_target *= self.loss_weights[name]
                    self.history[name].append(loss_target.item())
                    total_loss += loss_target

                self.history["total"].append(total_loss.item())
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                desc = f"Epoch[{epoch}/{n_epochs}] "
                if scheduler is not None:
                    desc += f"(lr: {scheduler.get_last_lr()[0]:.2e}) "
                    scheduler.step()

                if epoch % update_step == 0:
                    desc += ', '.join([f'{k.upper()}: {v[-1]:.2e}' for k, v in self.history.items()])
                    pbar.set_description(desc)
        return self.history

    @torch.no_grad()
    def predict(self, inputs):
        self.model.eval()
        pred = self.model(inputs)
        return pred.detach().cpu().squeeze().numpy()

    def show_history(self):
        fig, ax = plt.subplots(figsize=(5, 3))
        for name in self.history:
            epochs = range(1, len(self.history["total"]) + 1)
            ax.semilogy(epochs[::10], self.history[name][::10],
                        label=name.upper())
        ax.legend(loc="upper right")
        ax.grid(color='k', ls=':', lw=1)
        ax.set_xlabel("t")
        ax.set_ylabel("u(t)")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":

    pass