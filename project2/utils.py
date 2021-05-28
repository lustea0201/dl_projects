from torch import empty, randperm
from matplotlib.pyplot import figure, scatter, legend, show
from math import pi

D = 2
R_2 = empty(1).fill_(1/(2*pi)).squeeze()
CENTER = empty(2).fill_(0.5)

def generate_dataset(nSamples = 1000,):
    train_input = empty((nSamples*2, D)).uniform_(0, 1)
    radii_2 = ((train_input-CENTER)**2).sum(axis = 1)
    train_target = ((R_2-radii_2).sign().long()+1)/2


    train_target = train_target.unsqueeze(1)

    return train_input[:nSamples], train_target[:nSamples], train_input[nSamples:], train_target[nSamples:]

def show_dataset(X, y):
    figure(figsize=(6,6))
    pos = (y > 0)[:,0]
    scatter(X[pos,0], X[pos,1], c = 'green')
    scatter(X[~pos,0], X[~pos,1], c = 'red')
    n = 50
    scatter(CENTER[0]+empty(50).new_tensor([i/(n-1)*R_2.sqrt() for i in range(n)]), [CENTER[1]]*n, c = 'blue')
    legend(['1', '0', 'radius check'])
    show()


def train(model, X_train, y_train, y_test, X_test, nb_epochs, LS, lr, mini_batch_size, verbose = True):
    for epoch in range(nb_epochs):
        model.zero_grad()
        idx = randperm(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]
        for i in range(0, X_train.size(0), mini_batch_size):
            input_ = X_train.narrow(0, i, mini_batch_size)
            target = y_train.narrow(0, i, mini_batch_size)

            # forward pass
            out = model.forward(input_)
            loss = LS.forward(out, target)
            # backward pass
            grad_output = LS.backward()

            grad_input = model.backward(grad_output)

            model.step(lr)

        pred_train = model.forward(X_train) > 0.5

        target_train = y_train
        accuracy_train = (pred_train == target_train).sum()/len(target_train)
        pred_test = model.forward(X_test) > 0.5
        target_test = y_test
        accuracy_test = (pred_test == target_test).sum()/len(target_test)
        if verbose:
            print('Epoch {:d}: loss = {:.3f}, accuracy = {:.1f}% (train)/{:.1f}% (test))'.format(epoch, loss.item(),
                                                                                         accuracy_train*100,
                                                                                         accuracy_test*100))
    return accuracy_train.item(), accuracy_test.item()
