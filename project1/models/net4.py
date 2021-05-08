import torch
import torch.nn as nn
import torch.nn.functional as F

class Net4(nn.Module):
    def __init__(self, submodel, c1 = 32, c2 = 32, c3 = 64, h = 100, p = 0.3):
        super().__init__()
        self.digit_classifier = submodel
        self.digit_classifier.subnet = True
        self.digit_classifier.train()
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze(axis = 1)
        x2 = x[:, 1, :, :].unsqueeze(axis = 1)
        
        x1 = self.digit_classifier(x1)
        x2 = self.digit_classifier(x2)
        
        xx2 = self.fc3(x2)
        xx1 = self.fc3(x1)
        
        x = (xx2-xx1).squeeze()
        
        if not self.training:
            x = (x > 0).long()
        
        return x, x1, x2