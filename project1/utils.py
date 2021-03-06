import torch
import torch.nn as nn
import torch.nn.functional as F
    
def accuracy(model, input_, target, mini_batch_size, with_class = False):
    model.eval()
    nb_errors = 0
    for b in range(0, input_.size(0), mini_batch_size):
        if with_class: 
            output, _, _ = model(input_.narrow(0, b, mini_batch_size))
        else:
            output = model(input_.narrow(0, b, mini_batch_size))
            
        gt = target.narrow(0, b, mini_batch_size)
        nb_errors += (output != gt).sum().item()
    N = input_.shape[0]
    return 100-100*(N-nb_errors)/N


def train_model(model, train_input, train_target, test_input, test_target, nb_epochs, mini_batch_size, optimizer, criterion):

    train_accuracy = torch.empty(size=(1, nb_epochs))
    test_accuracy = torch.empty(size=(1, nb_epochs))
    train_loss = []
    
    for e in range(nb_epochs):
        model.train()
        epoch_loss = 0
        
        for b in range(0, train_input.size(0), mini_batch_size):
            input_ = train_input.narrow(0, b, mini_batch_size)
            output = model(input_)
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                
            epoch_loss += loss.item()

            model.zero_grad()
            
            loss.backward()
            optimizer.step()
        
        train_accuracy[0][e] = accuracy(model, train_input, train_target, mini_batch_size)
        test_accuracy[0][e] = accuracy(model, test_input, test_target, mini_batch_size)
        train_loss.append(epoch_loss)
    
    return train_accuracy, test_accuracy, train_loss

                
def train_model_double_objective(model, train_input, train_target, train_classes, test_input, test_target, test_classes, nb_epochs, mini_batch_size, optimizer, criterion, criterion2, beta = 1):
    
    train_accuracy = torch.empty(size=(1, nb_epochs))
    test_accuracy = torch.empty(size=(1, nb_epochs))
    train_loss = []
    
    for e in range(nb_epochs):
        model.train()
        epoch_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            input_ = train_input.narrow(0, b, mini_batch_size)
            output, output2, output3 = model(input_)
            
            loss = criterion(output, (train_target.narrow(0, b, mini_batch_size)).type_as(output))
            loss += beta*criterion2(output2, train_classes.narrow(0, b, mini_batch_size)[:,0])
            loss += beta*criterion2(output3, train_classes.narrow(0, b, mini_batch_size)[:,1])
                
            epoch_loss += loss.item()

            model.zero_grad()
            
            loss.backward()
            optimizer.step()        
        
        train_accuracy[0][e] = accuracy(model, train_input, train_target, mini_batch_size, with_class = True)
        test_accuracy[0][e] = accuracy(model, test_input, test_target, mini_batch_size, with_class = True)
        train_loss.append(epoch_loss)
    
    return train_accuracy, test_accuracy, train_loss
            

def accuracy_of_digit_class(model, input_, classes, mini_batch_size = 10):
  """ Evaluate the accuracy of the digit_classifier subnetwork of model """
  nb_errors = 0
  for b in range(0, input_.size(0), mini_batch_size): 
    _, out1, _ = model(input_.narrow(0, b, mini_batch_size))
    _, pred = torch.max(out1, dim=1)
    gt = classes.narrow(0, b, mini_batch_size)[:, 0]
    nb_errors += (pred != gt).sum().item()
  N = input_.shape[0]
  return 100*(N-nb_errors)/N

def dfs_freeze(model):
    """ Freeze all the weights and biases in the network, to avoid optimization """
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
        
def num_of_train_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def calculate_mean_std(train_accs, test_accs, train_losses):
    #train_losses = torch.tensor(train_losses)
    train_mean = torch.mean(train_accs, dim=0)
    test_mean = torch.mean(test_accs, dim=0)
    train_std = torch.zeros(size=train_mean.shape)
    train_std[-1] = torch.std(train_accs[:, -1])
    test_std = torch.zeros(size=test_mean.shape)
    test_std[-1] = torch.std(test_accs[:, -1])
    #loss_std, loss_mean = torch.std_mean(train_losses, dim=0)
    
    return train_mean, train_std, test_mean, test_std 
            

