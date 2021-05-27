from utils import generate_dataset, train
from module import Linear, Sequential, ReLU, Tanh, LossBCE, LossMSE
from torch import set_grad_enabled, manual_seed
set_grad_enabled(False) # REQUIRED
manual_seed(3) # For reproducibility

n_samples = 1000
X_train, y_train, X_test, y_test = generate_dataset(n_samples)

# Normalizing the data
mean, std = X_train.mean(), X_train.std()
X_train.sub_(mean).div_(std)
X_test.sub_(mean).div_(std)
input_dim, h1, h2, h3, output_dim = 2, 25, 25, 25, 1





activation = Tanh
init = 'xavier'
model = Sequential(Linear(input_dim, h1, init),
                   activation(),
                   Linear(h1, h2, init),
                   activation(),
                   Linear(h2, h3, init),
                   activation(),
                   Linear(h3, output_dim, init))
LS = LossMSE()
nb_epochs = 100
mini_batch_size = 50
lr = 1e-3/n_samples*mini_batch_size

train(model, X_train, y_train, y_test, X_test, nb_epochs, LS, lr, mini_batch_size)
