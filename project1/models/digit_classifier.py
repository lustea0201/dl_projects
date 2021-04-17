import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitClassifier(nn.Module):
    def __init__(self, out_h, subnet, c1 = 32, c2 = 32, c3 = 64, h = 100, p = 0.3):
        super().__init__()
        self.subnet = subnet
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=2)
        self.c3 = c3
        self.fc1 = nn.Linear(c3, h)
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(h, out_h)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, self.c3)))
        x = self.drop(x)
        x = self.fc2(x)
        
        if not self.training and not self.subnet:
            _, x = torch.max(x, dim=1)
        
        return x 