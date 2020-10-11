from bisect import bisect_right
from timm.optim import AdamW
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.rmsprop import RMSprop
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR
from torch.optim.lr_scheduler import _LRScheduler
from apex.optimizers import FusedAdam, FusedSGD

default_config = {

    "batch_size": 1,
    "type": "Adam",
    "weight_decay": 5e-4,
    "learning_rate": 1e-3,
    "schedule": {
        "type": "poly",
        "mode": "epoch",
        "epochs": 40,
        "params": {"max_iter": 50, "cycle": 1, "power": 0.8}
    }
}
class LRStepScheduler(_LRScheduler):
    def __init__(self, optimizer, steps, last_epoch=-1):
        self.lr_steps = steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        pos = max(bisect_right([x for x, y in self.lr_steps], self.last_epoch) - 1, 0)
        return [self.lr_steps[pos][1] if self.lr_steps[pos][0] <= self.last_epoch else base_lr for base_lr in self.base_lrs]


class PolyLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to poly learning rate policy
    """
    def __init__(self, optimizer, max_iter=90000, power=0.9, last_epoch=-1,cycle=False):
        self.max_iter = max_iter
        self.power = power
        self.cycle = cycle
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.last_epoch_div = (self.last_epoch + 1) % self.max_iter
        scale = (self.last_epoch + 1) // self.max_iter + 1.0 if self.cycle else 1
        return [(base_lr * ((1 - float(self.last_epoch_div) / self.max_iter) ** (self.power))) / scale for base_lr in self.base_lrs]

class ExponentialLRScheduler(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super(ExponentialLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= 0:
            return self.base_lrs
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]


def create_optimizer(model, optimizer_config=default_config, master_params=None):
    if optimizer_config.get("classifier_lr", -1) != -1:
        # Separate classifier parameters from all others
        net_params = []
        classifier_params = []
        for k, v in model.named_parameters():
            if not v.requires_grad:
                continue
            if k.find("encoder") != -1:
                net_params.append(v)
            else:
                classifier_params.append(v)
        params = [
            {"params": net_params},
            {"params": classifier_params, "lr": optimizer_config["classifier_lr"]},
        ]
    else:
        if master_params:
            params = master_params
        else:
            params = model.parameters()

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(params,
                              lr=optimizer_config["learning_rate"],
                              momentum=optimizer_config["momentum"],
                              weight_decay=optimizer_config["weight_decay"],
                              nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "FusedSGD":
        optimizer = FusedSGD(params,
                             lr=optimizer_config["learning_rate"],
                             momentum=optimizer_config["momentum"],
                             weight_decay=optimizer_config["weight_decay"],
                             nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(params,
                               lr=optimizer_config["learning_rate"],
                               betas=(0.9, 0.999),
                               eps=1e-07,
                               weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "FusedAdam":
        optimizer = FusedAdam(params,
                              lr=optimizer_config["learning_rate"],
                              weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "AdamW":
        optimizer = AdamW(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "RmsProp":
        optimizer = RMSprop(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))

    if optimizer_config["schedule"]["type"] == "step":
        scheduler = LRStepScheduler(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "clr":
        scheduler = CyclicLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "multistep":
        scheduler = MultiStepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "exponential":
        scheduler = ExponentialLRScheduler(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "poly":
        scheduler = PolyLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "constant":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif optimizer_config["schedule"]["type"] == "linear":
        def linear_lr(it):
            return it * optimizer_config["schedule"]["params"]["alpha"] + optimizer_config["schedule"]["params"]["beta"]

        scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    return optimizer, scheduler