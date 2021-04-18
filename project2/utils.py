from torch import tensor, rand, sqrt, linspace, stack
from matplotlib.pyplot import figure, scatter, legend, show
from math import pi

D = 2
R_2 = tensor(1/(2*pi))
CENTER = tensor([0.5, 0.5])

def generate_dataset(nSamples = 1000,):
    train_input = rand(nSamples*2, D)
    radii_2 = ((train_input-CENTER)**2).sum(axis = 1)
    train_target = ((R_2-radii_2).sign().long()+1)/2


    train_target = stack([train_target, train_target], axis = 1)
    train_target[:, 1] = 1 - train_target[:, 0]

    return train_input[:nSamples], train_target[:nSamples], train_input[nSamples:], train_target[nSamples:]

def show_dataset(X, y):
    figure(figsize=(6,6))
    pos = (y > 0)
    scatter(X[pos,0], X[pos,1], c = 'green')
    scatter(X[~pos,0], X[~pos,1], c = 'red')
    scatter(CENTER[0]+linspace(0, sqrt(R_2), 50), [CENTER[1]]*50, c = 'blue')
    legend(['1', '0', 'radius check'])
    show()
