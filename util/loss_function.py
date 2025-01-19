import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_args_parser
parser = get_args_parser()
args = parser.parse_args()
# Set device

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = args.num_classes
print("数据集类别数为：",num_classes)


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
