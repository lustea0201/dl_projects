import torch
import torch.nn as nn
import torch.nn.functional as F

class Net2(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(64, h)
        self.fc2 = nn.Linear(h, 10)
        self.fc3 = nn.Linear(10, 1)
        self.drop = nn.Dropout(p=0.3)
        
    def to_class_scores(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 64)))
        x = self.drop(x)
        x = self.fc2(x)
        
        return x 

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze(axis = 1)
        x2 = x[:, 1, :, :].unsqueeze(axis = 1)
        
        
        x1 = self.to_class_scores(x1)
        x2 = self.to_class_scores(x2)
        
        xx2 = self.fc3(x2)
        xx1 = self.fc3(x1)
        
        x = (xx2-xx1).squeeze()
        
        return x, x1, x2
