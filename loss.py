import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

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
        result = 0
        if torch.max(real_ix):
            result += self.loss(segment.view(segment.shape[0], -1)[real_ix],groundtruth.data.view(groundtruth.shape[0], -1)[real_ix])
        if torch.max(fake_ix):
            result += self.loss(segment.view(segment.shape[0], -1)[fake_ix],groundtruth.data.view(groundtruth.shape[0], -1)[fake_ix])
        if torch.isnan(result):
            pdb.set_trace()
        return result
        # return 1.0 * (torch.sum(real_ix) > 0) * 3 * self.loss(segment.view(segment.shape[0], -1)[real_ix],
        #                  groundtruth.data.view(groundtruth.shape[0], -1)[real_ix]) + \
        #        1.0 * (torch.sum(fake_ix) > 0) * self.loss(segment.view(segment.shape[0], -1)[fake_ix],
        #                  groundtruth.data.view(groundtruth.shape[0], -1)[fake_ix])


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
        pt = pt.clamp(min=0.00001, max=1.0-0.00001)
        result = 0
        if torch.max(real_ix):
            result += 3 * self.loss_func(pt[real_ix],target[real_ix])
        if torch.max(fake_ix):
            result += self.loss_func(pt[fake_ix],target[fake_ix])
        if torch.isnan(result):
            pdb.set_trace()
        return result

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = input
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


