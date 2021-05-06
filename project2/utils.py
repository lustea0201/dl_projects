from torch import linspace, empty
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
    pos = (y > 0)
    scatter(X[pos,0], X[pos,1], c = 'green')
    scatter(X[~pos,0], X[~pos,1], c = 'red')
    scatter(CENTER[0]+linspace(0, R_2.sqrt(), 50), [CENTER[1]]*50, c = 'blue')
    legend(['1', '0', 'radius check'])
    show()
