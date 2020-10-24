import torch
import torch.nn as nn
class ActivationLoss(nn.Module):
    def __init__(self):
        super(ActivationLoss, self).__init__()

    def forward(self, zero, one, labels):
        loss_act = torch.abs(one - labels.data) + torch.abs(zero - (1.0 - labels.data))
        return 1 / labels.shape[0] * loss_act.sum()


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, reconstruction, groundtruth):
        return self.loss(reconstruction, groundtruth.data)


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, segment, groundtruth,real_ix,fake_ix):
        return 2 * self.loss(segment.view(segment.shape[0], -1)[real_ix],
                         groundtruth.data.view(groundtruth.shape[0], -1)[real_ix]) + \
               self.loss(segment.view(segment.shape[0], -1)[fake_ix],
                         groundtruth.data.view(groundtruth.shape[0], -1)[fake_ix])


class SegFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction="elementwise_mean"):
        super(SegFocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def smooth(self,inputs):
        ix_1 = (inputs ==1)
        ix_0 = (inputs ==0)
        inputs[ix_1] -= 1e-6
        inputs[ix_0] += 1e-6
        return inputs

    def loss_func(self,pt,target):
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == "elementwise_mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss

    def forward(self, pt, target,real_ix,fake_ix):
        pt = pt.reshape(pt.shape[0],-1)
        target = target.reshape(target.shape[0],-1)
        return 1.0 * (torch.sum(real_ix) > 0) * 4 * self.loss_func(pt[real_ix],target[real_ix]) + \
               1.0 * (torch.sum(fake_ix) > 0) * self.loss_func(pt[fake_ix],target[fake_ix])

