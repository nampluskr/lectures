import torch
from copy import deepcopy
import sys
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_model = None
        self.triggered = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            model.load_state_dict(self.best_model)
            self.triggered = True
            return True

        return False


class Trainer:
    def __init__(self, model, optimizer, loss_fn, metrics):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = {"loss": loss_fn}
        if metrics is not None:
            self.metrics.update(metrics)
        self.device = next(model.parameters()).device

    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {name: func(y_pred, y).item() for name, func in self.metrics.items()}

    @torch.no_grad()
    def test_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.model(x)
        return {name: func(y_pred, y).item() for name, func in self.metrics.items()}

    def fit(self, train_loader, n_epochs, valid_loader=None, step_size=1,
            scheduler=None, stopper=None):
        history = {name: [] for name in self.metrics}
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in self.metrics})

        for epoch in range(1, n_epochs + 1):
            cur_epoch = str(epoch).rjust(len(str(n_epochs)), ' ')
            cur_epoch = f"[{cur_epoch}/{n_epochs}]"

            ## Training
            self.model.train()
            res = {name: 0 for name in self.metrics}
            with tqdm(train_loader, leave=False, file=sys.stdout,
                      dynamic_ncols=True, ascii=True) as pbar:
                for i, (x, y) in enumerate(pbar):
                    res_step = self.train_step(x, y)

                    desc = ""
                    for name in self.metrics:
                        res[name] += res_step[name]
                        if name == "loss":
                            desc += f" {name}: {res[name]/(i+1):.2e}"
                        else:
                            desc += f" {name}: {res[name]/(i+1):.3f}"

                    pbar.set_description(cur_epoch + desc)

            for name in self.metrics:
                history[name].append(res[name]/len(train_loader))

            if scheduler is not None:
                scheduler.step()

            if valid_loader is None:
                if epoch % step_size == 0:
                    print(cur_epoch + desc)
                continue

            ## Validation
            res = self.evaluate(valid_loader)
            val_desc = ""
            for name in self.metrics:
                history[f"val_{name}"].append(res[name])
                if name == "loss":
                    val_desc += f" val_{name}: {res[name]:.2e}"
                else:
                    val_desc += f" val_{name}: {res[name]:.3f}"

            if epoch % step_size == 0:
                print(cur_epoch + desc + " |" + val_desc)

            ## Early Stopping
            if stopper is not None:
                val_loss = history["val_loss"][-1]
                stopper.step(val_loss, self.model)
                if stopper.triggered:
                    print(f">> Early stopped! (best_loss: {stopper.best_loss:.3f})")
                    break

        return history

    def evaluate(self, test_loader):
        self.model.eval()
        res = {name: 0 for name in self.metrics}
        with tqdm(test_loader, desc="Evaluation", leave=False, file=sys.stdout,
                  dynamic_ncols=True, ascii=True) as pbar:
            for x, y in pbar:
                res_step = self.test_step(x, y)

                for name in self.metrics:
                    res[name] += res_step[name]

        for name in self.metrics:
            res[name] /= len(test_loader)
        return res


class AETrainer(Trainer):
    def train_step(self, x, y):
        x = x.to(self.device)
        x_pred = self.model(x)
        loss = self.loss_fn(x_pred, x)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {name: func(x_pred, x).item() for name, func in self.metrics.items()}

    @torch.no_grad()
    def test_step(self, x, y):
        x = x.to(self.device)
        x_pred = self.model(x)
        return {name: func(x_pred, x).item() for name, func in self.metrics.items()}


class VAETrainer(Trainer):
    def train_step(self, x, y):
        x = x.to(self.device)
        x_pred, mu, logvar = self.model(x)
        loss = self.loss_fn(x_pred, x, mu, logvar)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        res = {"loss": loss.item()}
        res.update({name: func(x_pred, x).item() 
                    for name, func in self.metrics.items() if name != "loss"})
        return res

    @torch.no_grad()
    def test_step(self, x, y):
        x = x.to(self.device)
        x_pred, mu, logvar = self.model(x)
        loss = self.loss_fn(x_pred, x, mu, logvar)

        res = {"loss": loss.item()}
        res.update({name: func(x_pred, x).item() 
                    for name, func in self.metrics.items() if name != "loss"})
        return res


if __name__ == "__main__":

    pass
