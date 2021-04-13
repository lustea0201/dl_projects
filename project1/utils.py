import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 4)

GREATER = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype = torch.uint8)

LESSOREQUAL = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.uint8)

def show_pair(X, y, i):
    C = X[i].max().item()
    symbol = C*(LESSOREQUAL if y[i] else GREATER)
    plt.imshow(torch.cat([X[i][0], symbol, X[i][1]], axis = 1))
    plt.axis("off")
    
   



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
    
def accuracy(model, input_, target, mini_batch_size, with_class = False):
    nb_errors = 0
    for b in range(0, input_.size(0), mini_batch_size):
        if with_class: 
            output, _, _ = model(input_.narrow(0, b, mini_batch_size))
        else:
            output = model(input_.narrow(0, b, mini_batch_size))
            
        pred = (output > 0).long()
        gt = target.narrow(0, b, mini_batch_size)
        nb_errors += (pred != gt).sum().item()
    N = input_.shape[0]
    return 100*(N-nb_errors)/N



def train_model(model, train_input, train_target, test_input, test_target, nb_epochs, mini_batch_size, optimizer, criterion, verbose = True):
    for e in range(nb_epochs):
        epoch_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            input_ = train_input.narrow(0, b, mini_batch_size)
            output = model(input_)
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                
            epoch_loss += loss.item()

            model.zero_grad()
            
            loss.backward()
            optimizer.step()
                    
        train_acc = accuracy(model, train_input, train_target, mini_batch_size)
        test_acc = accuracy(model, test_input, test_target, mini_batch_size)
        
        if verbose:
            print('Epoch {:d}: loss {:.3f} / train accuracy {:.1f}%, test accuracy {:.1f}'.format(
                e, epoch_loss, train_acc, test_acc))
def train_model_double_objective(model, train_input, train_target, train_classes, test_input, test_target, test_classes, nb_epochs, mini_batch_size, optimizer, criterion, criterion2, beta = 1, verbose = True):
    for e in range(nb_epochs):
        epoch_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            input_ = train_input.narrow(0, b, mini_batch_size)
            output, output2, output3 = model(input_)
            
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            loss += beta*criterion2(output2, train_classes.narrow(0, b, mini_batch_size)[:,0])
            loss += beta*criterion2(output3, train_classes.narrow(0, b, mini_batch_size)[:,1])
                
            epoch_loss += loss.item()

            model.zero_grad()
            
            loss.backward()
            optimizer.step()
                    
        train_acc = accuracy(model, train_input, train_target, mini_batch_size, with_class = True)
        test_acc = accuracy(model, test_input, test_target, mini_batch_size, with_class = True)
        
        if verbose:
            print('Epoch {:d}: loss {:.3f} / train accuracy {:.1f}%, test accuracy {:.1f}'.format(
                e, epoch_loss, train_acc, test_acc))
            
            

