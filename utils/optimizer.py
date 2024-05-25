import torch
from bisect import bisect_right

def make_optimizer_with_triplet(model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 0.00035
        weight_decay = 5e-4
        if "bias" in key:
            lr = lr * 1.0
            weight_decay = 5e-4
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, "SGD")(params, momentum=0.9)

    return optimizer


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones          # (40, 70)
        self.gamma = gamma        # 0.1
        self.warmup_factor = warmup_factor    # 0.01
        self.warmup_iters = warmup_iters      # 0
        self.warmup_method = warmup_method    # linear
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]