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
        grad_input = grad_output *  self.f_prime(self.input)
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
        self.b = torch.empty(output_dim).normal_(0, sigma)
        self.zero_grad()

    def zero_grad(self):
        self.grad_W = torch.zeros(self.W.shape)
        self.grad_b = torch.zeros(self.b.shape)


    def forward(self, input_):
        """ input_: (B x D_input)"""

        self.input2 = input_
        output = (torch.mm(self.W, input_.T) + self.b.unsqueeze(1)).T

        return output
        
    
    def backward(self, grad_output):
        input_ = self.input2
    
        grad_input = torch.mm(grad_output, self.W)
        self.grad_W += torch.mm(grad_output.T, input_)
        self.grad_b += grad_output.T.sum(axis = 1)

        return grad_input


    def param(self):
        return [(self.W, self.grad_W), (self.b, self.grad_b)]
