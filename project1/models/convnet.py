import torch
from torch import nn

class ConvNet(nn.Module):

  def __init__(self, nb_channels, kernel_size = 3):
    super().__init__()
    self.nb_channels = nb_channels

    # 2 x 14 x 14
    self.conv1 = nn.Conv2d(in_channels=2, 
                           out_channels=nb_channels,
                           kernel_size=kernel_size,
                           padding = (kernel_size - 1) // 2)
    # nb_channels x 14 x 14
    self.bn1 = nn.BatchNorm2d(nb_channels)
    # nb_channels x 14 x 14
    self.pool1 = nn.MaxPool2d(kernel_size=2)

    # nb_channels x 7 x 7
    self.lin1 = nn.Linear(in_features=nb_channels*7*7,
                          out_features=1024
                         )
    # 1024
    self.lin2 = nn.Linear(in_features=1024,
                          out_features=1
                         )
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
  

  def forward(self, x):
    x = self.pool1(self.bn1(self.conv1(x)))
    x = x.view(-1, self.nb_channels*7*7)
    x = self.relu(self.lin1(x))
    x = self.sigmoid(self.lin2(x))
    if not self.training:
      x = 1 * (x > 0.5)

    return x.view(-1)
