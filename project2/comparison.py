from utils import generate_dataset, train
from module import Linear, Sequential, ReLU, Tanh, LossBCE, LossMSE
from torch import set_grad_enabled, manual_seed
from math import sqrt
set_grad_enabled(False) # REQUIRED
manual_seed(3) # For reproducibility

n_samples = 1000
X_train, y_train, X_test, y_test = generate_dataset(n_samples)

# Normalizing the data
mean, std = X_train.mean(), X_train.std()
X_train.sub_(mean).div_(std)
X_test.sub_(mean).div_(std)

# Dimensions of the network
input_dim, h1, h2, h3, output_dim = 2, 25, 25, 25, 1

# Training parameters
nb_epochs = 100
mini_batch_size = 50
lr = 1e-3/n_samples*mini_batch_size


activations = [ReLU, Tanh]
activation_names = ["ReLU", "Tanh"]
losses = [LossMSE(), LossBCE()]
loss_names = ["MSE", "BCE"]

for l, LS in enumerate(losses):
	for a, activation in enumerate(activations):
		for init in ['normal', 'xavier', 'he']:
			accs_train, accs_test = [], []
			for i in range(100):
				model = Sequential(Linear(input_dim, h1, init),
	                   activation(),
	                   Linear(h1, h2, init),
	                   activation(),
	                   Linear(h2, h3, init),
	                   activation(),
					   Linear(h3, output_dim, init))

				acc_train, acc_test = train(model, X_train, y_train, y_test, X_test, nb_epochs, LS, lr, mini_batch_size, False)
				accs_train.append(100*acc_train)
				accs_test.append(100*acc_test)

			mean_train = sum(accs_train)/len(accs_train)
			mean_test = sum(accs_test)/len(accs_test)

			std_train = sqrt(sum(pow(x-mean_train,2) for x in accs_train) / (len(accs_train)-1))
			std_test = sqrt(sum(pow(x-mean_test,2) for x in accs_test) / (len(accs_test)-1))

			print('{:s}/{:s}/{:s}: {:.2f} +- {:.2f} (train) & {:.2f} +- {:.2f} (test)'.format(loss_names[l], activation_names[a], init, mean_train, std_train, mean_test, std_test))
