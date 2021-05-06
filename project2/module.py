from torch import empty

class Module(object):
    def __init__(self):
        self.trainable = False
        
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
    
class LossMSE(Module):
    def forward(self , prediction, target):
        error = target - prediction
        self.error = error
        self.n = len(prediction)
        loss = error.pow(2).sum()/self.n
        
        return loss
        
    def backward(self):
        return -2*self.error/self.n
    
    def param(self):
        return []


class Linear(Module):
    def __init__(self, input_dim, output_dim, sigma = 1):
        self.W = empty((output_dim, input_dim)).normal_(0, sigma)
        self.b = empty(output_dim).normal_(0, sigma)
        self.zero_grad()
        self.trainable = True

    def zero_grad(self):
        self.grad_W = empty(self.W.shape).zero_() 
        self.grad_b = empty(self.b.shape).zero_()


    def forward(self, input_):
        """ input_: (B x D_input)"""

        self.input2 = input_
        output = (self.W.mm(input_.T) + self.b.unsqueeze(1)).T

        return output
        
    
    def backward(self, grad_output):
        input_ = self.input2
    
        grad_input = grad_output.mm(self.W)
        self.grad_W += grad_output.T.mm(input_)
        self.grad_b += grad_output.T.sum(axis = 1)

        return grad_input


    def param(self):
        return [(self.W, self.grad_W), (self.b, self.grad_b)]
    
class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
        
    def forward(self , input_):
        output = input_
        for layer in self.layers:
            output = layer.forward(output)
        return output
        
    def backward(self , grad_output):
        grad_input = grad_output
        for layer in reversed(self.layers):
            grad_input = layer.backward(grad_input)
        return grad_input
        
    def zero_grad(self):
        for layer in self.layers:
            if layer.trainable:
                layer.zero_grad()
            
    def param(self):
        return []
