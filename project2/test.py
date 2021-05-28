from utils import generate_dataset, train
from module import Linear, Sequential, ReLU, Tanh, LossBCE, LossMSE
from torch import set_grad_enabled, manual_seed

set_grad_enabled(False) # REQUIRED
manual_seed(0) # For reproducibility

n_samples = 1000
X_train, y_train, X_test, y_test = generate_dataset(n_samples)

# Normalizing the data
mean, std = X_train.mean(), X_train.std()
X_train.sub_(mean).div_(std)
X_test.sub_(mean).div_(std)

# Choosing the network architecure
input_dim, h1, h2, h3, output_dim = 2, 25, 25, 25, 1 # Dimensions
activation = Tanh
init = 'xavier' # Weight initialization strategy for the fully connected layers
model = Sequential(Linear(input_dim, h1, init),
                   activation(),
                   Linear(h1, h2, init),
                   activation(),
                   Linear(h2, h3, init),
                   activation(),
                   Linear(h3, output_dim, init))
LS = LossMSE() # Loss function

# Training parameters
nb_epochs = 100
mini_batch_size = 50
lr = 1e-3

train_err, test_err = train(model, X_train, y_train, y_test, X_test, nb_epochs, LS, lr, mini_batch_size, True)
print('Final error: {:.2f}% on training set, {:.2f}% on testing set'.format(100*train_err, 100*test_err))
