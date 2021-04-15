import torch
import torch.nn as nn
import torch.nn.functional as F
from models.digit_classifier import DigitClassifier

class Net2(nn.Module):
    def __init__(self, c1 = 32, c2 = 32, c3 = 64, h = 100, p = 0.3):
        super().__init__()
        self.digit_classifier = DigitClassifier(10, c1, c2, c3, h, p)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze(axis = 1)
        x2 = x[:, 1, :, :].unsqueeze(axis = 1)
        
        x1 = self.digit_classifier(x1)
        x2 = self.digit_classifier(x2)
        
        xx2 = self.fc3(x2)
        xx1 = self.fc3(x1)
        
        x = (xx2-xx1).squeeze()
        
        return x, x1, x2
    

