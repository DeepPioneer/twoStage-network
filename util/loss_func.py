import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# Set device

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 重新加权损失和focal loss损失的结合
class ReweightedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, class_weights=None):
        super(ReweightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)

        return focal_loss.mean()

# focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss

# 加权损失
class ReweightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(ReweightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights)
        return ce_loss

class DynamicWeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=5):
        super(DynamicWeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Calculate the weight for each class dynamically based on the current batch
        class_counts = torch.bincount(targets, minlength=self.num_classes)
        # class_weights = 1.0 / (class_counts.float() + 1e-6)
        # class_weights /= class_weights.sum()
        class_counts = class_counts / class_counts.sum()
        class_weights = 1 - class_counts

        if self.alpha is not None:
            alpha = self.alpha * class_weights
        else:
            alpha = class_weights

        # Get class weights for each sample in the batch
        alpha_for_each_sample = alpha[targets]

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        pt = torch.exp(-BCE_loss)
        F_loss = alpha_for_each_sample * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean()

# For MNIST dataset we difine classes to 10
classes = 5

class GuidedComplementEntropy(nn.Module):

    def __init__(self, alpha):
        super(GuidedComplementEntropy, self).__init__()
        self.alpha = alpha

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        self.classes = classes
        yHat = F.softmax(yHat, dim=1)
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
        # avoiding numerical issues (second)
        guided_factor = (Yg + 1e-7) ** self.alpha
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (third)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        guided_output = guided_factor.squeeze() * torch.sum(output, dim=1)
        loss = torch.sum(guided_output)
        loss /= float(self.batch_size)
        loss /= math.log(float(self.classes))
        return loss

class ComplementEntropy(nn.Module):

    def __init__(self):
        super(ComplementEntropy, self).__init__()

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        self.classes = classes
        yHat = F.softmax(yHat, dim=1)
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.classes)
        return loss

# Define class weights if needed
class_weights = torch.tensor([1.0, 2.0, 1.5, 1.0, 1.0])

# 义损失函数（使用自定义的重加权损失）
reweighted_loss = ReweightedCrossEntropyLoss(class_weights=class_weights)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, class_num=5, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
