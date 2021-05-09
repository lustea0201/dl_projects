import torch
from torch import nn

class ConvNet(nn.Module):

  def __init__(self, nb_channels, kernel_size = 3, weight_sharing = False):
    super().__init__()
    self.nb_channels = nb_channels
    self.kernel_size = kernel_size
    self.weight_sharing = weight_sharing

    # 2 x 16 x 16
    self.conv1 = nn.Conv2d(in_channels=2, 
                           out_channels=nb_channels,
                           kernel_size=kernel_size,
                           padding = (kernel_size - 1) // 2)
    # nb_channels x 16 x 16
    self.bn1 = nn.BatchNorm2d(nb_channels)
    # nb_channels x 16 x 16
    self.pool1 = nn.MaxPool2d(kernel_size=2)
    
    # nb_channels x 8 x 8
    self.conv2 = nn.Conv2d(in_channels=nb_channels, 
                           out_channels=nb_channels,
                           kernel_size=kernel_size,
                           padding = (kernel_size - 1) // 2)
    # nb_channels x 8 x 8
    self.bn2 = nn.BatchNorm2d(nb_channels)
    # nb_channels x 8 x 8
    self.pool2 = nn.MaxPool2d(kernel_size=2)
    
    if not self.weight_sharing:
        # nb_channels x 4 x 4
        self.conv3 = nn.Conv2d(in_channels=nb_channels, 
                               out_channels=nb_channels,
                               kernel_size=kernel_size,
                               padding = (kernel_size - 1) // 2)
        # nb_channels x 4 x 4
        self.bn3 = nn.BatchNorm2d(nb_channels)
        # nb_channels x 4 x 4
        self.pool3 = nn.MaxPool2d(kernel_size=2)

    # nb_channels x 2 x 2
    self.lin1 = nn.Linear(in_features=nb_channels*2*2,
                          out_features=2*nb_channels*2*2)
    # 2*nb_channels*2*2
    self.lin2 = nn.Linear(in_features=2*nb_channels*2*2,
                          out_features=1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    

  def forward(self, x):
    x = self.pool1(self.bn1(self.conv1(x)))
    x = self.pool2(self.bn2(self.conv2(x)))
    if self.weight_sharing:
        x = self.pool2(self.bn2(self.conv2(x)))
    else:
        x = self.pool3(self.bn3(self.conv3(x)))
    x = x.view(-1, self.nb_channels*2*2)
    x = self.relu(self.lin1(x))
    x = self.sigmoid(self.lin2(x))
    if not self.training:
      x = 1 * (x > 0.5)

    return x.view(-1)
