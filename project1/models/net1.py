import torch
import torch.nn as nn
import torch.nn.functional as F
from models.digit_classifier import DigitClassifier

class Net1(nn.Module):
    def __init__(self, weight_sharing=False, c1 = 32, c2 = 32, c3 = 64, h = 100, p = 0.3):
        super().__init__()
        self.weight_sharing = weight_sharing
        self.digit_classifier1 = DigitClassifier(1, True, c1, c2, c3, h, p)
        if not self.weight_sharing:
            self.digit_classifier2 = DigitClassifier(1, True, c1, c2, c3, h, p)

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze(axis = 1)
        x2 = x[:, 1, :, :].unsqueeze(axis = 1)

        x1 = self.digit_classifier1(x1)
        if self.weight_sharing:
            x2 = self.digit_classifier1(x2)
        else:
            x2 = self.digit_classifier2(x2)

        x = (x2-x1).squeeze()
        
        if not self.training:
            x = (x > 0).long()

        return x
