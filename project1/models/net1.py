import torch
import torch.nn as nn
import torch.nn.functional as F
from models.digit_classifier import DigitClassifier

class Net1(nn.Module):
    def __init__(self, c1 = 32, c2 = 32, c3 = 64, h = 100, p = 0.3):
        super().__init__()
        self.digit_classifier = DigitClassifier(1, c1, c2, c3, h, p)

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze(axis = 1)
        x2 = x[:, 1, :, :].unsqueeze(axis = 1)

        x1 = self.digit_classifier(x1)
        x2 = self.digit_classifier(x2)

        x = (x2-x1).squeeze()

        return x
