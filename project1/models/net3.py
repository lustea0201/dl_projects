import torch
import torch.nn as nn
import torch.nn.functional as F
from models.digit_classifier import DigitClassifier
#from models.resnetblock import ResNetBlock

class ResNetBlock(nn.Module):
    def __init__(self, nb_channels = 1, kernel_size = 3, batch_normalization = True, hidden_channel_1 = 1, hidden_channel_2 = 10 ):
        super().__init__()

        self.conv1 = nn.Conv2d(hidden_channel_1, hidden_channel_2,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(hidden_channel_2)

        self.conv2 = nn.Conv2d(hidden_channel_2, hidden_channel_1,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn2 = nn.BatchNorm2d(hidden_channel_1)
        
        self.batch_normalization = batch_normalization

    def forward(self, x): 
        
        y = self.conv1(x)
        
        if self.batch_normalization: 
            y = self.bn1(y)
            
        y = F.relu(y)
        y = self.conv2(y)
        
        if self.batch_normalization: 
            y = self.bn2(y)
            
        y = y + x
        y = F.relu(y)
        return y
    
class Net3(nn.Module):
    def __init__(self, nb_residual_blocks, pretrained_submodel = None, c1 = 32, c2 = 32, c3 = 64, h = 100, p = 0.3, hidden_channel_1 = 1, hidden_channel_2 = 10):
        super().__init__()
        if pretrained_submodel is None:
            self.digit_classifier = DigitClassifier(10, True, c1, c2, c3, h, p)
        else:
            self.digit_classifier = pretrained_submodel
            self.digit_classifier.subnet = True
            self.digit_classifier.train()
        #self.digit_classifier = DigitClassifier(10, True, c1, c2, c3, h, p)
        self.fc3 = nn.Linear(10, 1)
        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(1,3, True, hidden_channel_1 , hidden_channel_2 )
              for _ in range(nb_residual_blocks))
        )
        
    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze(axis = 1)
        x2 = x[:, 1, :, :].unsqueeze(axis = 1)
        
        x1 = self.resnet_blocks(x1)
        x2 = self.resnet_blocks(x2)
        
        x1 = self.digit_classifier(x1)
        x2 = self.digit_classifier(x2)
        
        xx2 = self.fc3(x2)
        xx1 = self.fc3(x1)
        
        x = (xx2-xx1).squeeze()
        
        if not self.training:
            x = (x > 0).long()
        
        return x, x1, x2  