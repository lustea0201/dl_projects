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
    return 100*(N-nb_errors)/N


def train_model(model, train_input, train_target, test_input, test_target, nb_epochs, mini_batch_size, optimizer, criterion):

    train_accuracy, test_accuracy, train_loss = [], [], []
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
        
        train_acc = accuracy(model, train_input, train_target, mini_batch_size)
        test_acc = accuracy(model, test_input, test_target, mini_batch_size)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        train_loss.append(epoch_loss)
        #print('Epoch {:d}: loss {:.3f} / train accuracy {:.1f}%, test accuracy {:.1f}'.format(
        #    e, epoch_loss, train_acc, test_acc))
    
    return train_accuracy, test_accuracy, train_loss

                
def train_model_double_objective(model, train_input, train_target, train_classes, test_input, test_target, test_classes, nb_epochs, mini_batch_size, optimizer, criterion, criterion2, beta = 1):
    
    train_accuracy, test_accuracy, train_loss = [], [], []
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
        
        train_acc = accuracy(model, train_input, train_target, mini_batch_size, with_class = True)
        test_acc = accuracy(model, test_input, test_target, mini_batch_size, with_class = True)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        train_loss.append(epoch_loss)
        #print('Epoch {:d}: loss {:.3f} / train accuracy {:.1f}%, test accuracy {:.1f}'.format(
        #    e, epoch_loss, train_acc, test_acc))
    
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
    nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {:d} trainable parameters'.format(nParams))
    
    
def visualize_accuracy(train_accs, test_accs, train_losses):
    train_accs = torch.tensor(train_accs)
    test_accs = torch.tensor(test_accs)
    train_losses = torch.tensor(train_losses)
    train_std, train_mean = torch.std_mean(train_accs, dim=0)
    test_std, test_mean = torch.std_mean(test_accs, dim=0)
    #loss_std, loss_mean = torch.std_mean(train_losses, dim=0)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 7))
    plt.xlabel('epochs')
    plt.ylabel('accuracy(%)')
    nb_epochs = train_accs.shape[1]
    plt.xticks(range(nb_epochs))
    plt.errorbar(range(nb_epochs), train_mean, yerr=train_std, fmt='b-o', label="train")
    plt.errorbar(range(nb_epochs), test_mean, yerr=test_std, fmt='g-o', label="test")
    #plt.errorbar(range(nb_epochs), loss_mean, yerr=loss_std, fmt='r-o')
    plt.legend(loc="upper right")
    plt.show()
            
            

