import torch

class Module(object):
    def forward(self , *input):
        raise NotImplementedError
    def backward(self , *gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []


class Activation(Module):
    def forward(self, input_):
        output = self.f(input_)
        self.input = input_
        return output

    def backward(self, grad_output):
        grad_input = grad_output *  self.f_prime(self.input_)
        return grad_input



class ReLU(Activation):
    @staticmethod
    def f(x):
        return x.clamp(min = 0)

    @staticmethod
    def f_prime(x):
        return (x.sign()+1)/2


class Tanh(Activation):
    @staticmethod
    def f(x):
        return x.tanh()

    @staticmethod
    def f_prime(x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)


class Linear(Module):
    def __init__(self, input_dim, output_dim, sigma = 1):
        self.W = torch.empty((output_dim, input_dim)).normal_(0, sigma)
        self.b = torch.empty(output_dim, 1).normal_(0, sigma)
        self.zero_grad()

    def zero_grad(self):
        self.grad_W = torch.zeros(self.W.shape)
        self.grad_b = torch.zeros(self.b.shape)



    def forward(self, input_):

        self.input = input_
        output = self.W.mm(input_) + self.b

        return output

    def backward(self, grad_output):
        input_ = self.input

        grad_input = torch.mm(self.W.T, grad_output)
        self.grad_W += torch.mm(grad_output, self.input.T)
        self.grad_b += grad_output

        return grad_input


    def param(self):
        return [(self.W, self.grad_W), (self.b, self.grad_b)]
