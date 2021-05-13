from utils import generate_dataset
from module import Linear, Sequential, ReLU, Tanh, LossBCE, LossMSE
from torch import set_grad_enabled, manual_seed
set_grad_enabled(False) # REQUIRED
manual_seed(3)

n_samples = 1000
X_train, y_train, X_test, y_test = generate_dataset(n_samples)

input_dim, h1, h2, h3, output_dim = 2, 25, 25, 25, 1



def train(model, nb_epochs, LS, lr, mini_batch_size):
    for epoch in range(nb_epochs):
        model.zero_grad()
        for i in range(0, X_train.size(0), mini_batch_size):
            input_ = X_train.narrow(0, i, mini_batch_size)
            target = y_train.narrow(0, i, mini_batch_size)

            # forward pass
            out = model.forward(input_)
            loss = LS.forward(out, target)
            # backward pass
            grad_output = LS.backward()

            grad_input = model.backward(grad_output)

            for layer in model.layers:
                if layer.trainable:
                    layer.W -= lr*layer.grad_W
                    layer.b -= lr*layer.grad_b

        pred_train = model.forward(X_train) > 0.5

        target_train = y_train
        accuracy_train = (pred_train == target_train).sum()/len(target_train)
        pred_test = model.forward(X_test) > 0.5
        target_test = y_test
        accuracy_test = (pred_test == target_test).sum()/len(target_test)
        print('Epoch {:d}: loss = {:.3f}, accuracy = {:.1f}% (train)/{:.1f}% (test))'.format(epoch, loss.item(),
                                                                                             accuracy_train*100,
                                                                                             accuracy_test*100))
activation = Tanh
model = Sequential(Linear(input_dim, h1),
                   activation(),
                   Linear(h1, h2),
                   activation(),
                   Linear(h2, h3),
                   activation(),
                   Linear(h3, output_dim))
LS = LossMSE()
nb_epochs = 100
mini_batch_size = 50
lr = 1e-3/n_samples*mini_batch_size # 1e-2 for MSE

train(model, nb_epochs, LS, lr, mini_batch_size)



activation = ReLU
model = Sequential(Linear(input_dim, h1),
                   activation(),
                   Linear(h1, h2),
                   activation(),
                   Linear(h2, h3),
                   activation(),
                   Linear(h3, output_dim))
LS = LossBCE()
nb_epochs = 100
mini_batch_size = 50
lr = 1e-3/n_samples*mini_batch_size # 1e-2 for MSE

train(model, nb_epochs, LS, lr, mini_batch_size)
