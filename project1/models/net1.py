import torch
import torch.nn as nn
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self, c1 = 32, c2 = 32, c3 = 64, h2 = 100, p = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=2)
        self.h1 = c3
        self.fc1 = nn.Linear(self.h1, h2)
        self.fc2 = nn.Linear(h2, 1)
        self.drop = nn.Dropout(p)
        
    def element_forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, self.h1)))
        x = self.drop(x)
        x = self.fc2(x)
        
        return x 

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze(axis = 1)
        x2 = x[:, 1, :, :].unsqueeze(axis = 1)
        
        x1 = self.element_forward(x1)
        x2 = self.element_forward(x2)
        
        x = (x2-x1).squeeze()
        
        return x
