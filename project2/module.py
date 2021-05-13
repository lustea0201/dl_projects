from torch import empty

class Module(object):
    """ Base class. """
    def __init__(self):
        self.trainable = False

    def forward(self , *input_):
        raise NotImplementedError
    def backward(self , *gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []


class Activation(Module):
    """ Generic class for an activation function. 
    """
    
    def forward(self, input_):
        """ Performs the forward pass for a generic activation function. 
        Stores the input_ for the backward pass and call the function itself. 
        
        Parameters
        ----------
        input_: torch.Tensor (any shape, typically (batch_size, dimension))
            The input to the activation function
            
            
        Returns
        -------
        output: torch.Tensor(same shape as input_)
            The output after passing through the activation
        """
        
        output = self.f(input_)
        self.input = input_
        
        return output

    def backward(self, grad_output):
        """ Performs the backward pass for a generic activation function. 
         
        
        Parameters
        ----------
        grad_output: torch.Tensor (same shape as self.input)
            The gradient at the output of the activation function
            
            
        Returns
        -------
        grad_input: torch.Tensor(same shape as self.input)
            The gradient at the input of the activation function
        """
        
        grad_input = grad_output *  self.f_prime(self.input)
        
        return grad_input
    
    @staticmethod
    def f(self , *input_):
        raise NotImplementedError
    
    @staticmethod
    def f_prime(self , *input_):
        raise NotImplementedError



class ReLU(Activation):
    """ Rectified Linear Unit class.
    Overrides f and f_prime.
    """
    
    @staticmethod
    def f(x):
        """ Applies the ReLU function to x.
         
        
        Parameters
        ----------
        x: torch.Tensor (any shape, typically (batch_size, dimension))
            The input to the ReLU function
            
            
        Returns
        -------
        y: torch.Tensor(same shape as x)
            The result of applying ReLU to x
        """
        
        y = x.clamp(min = 0)
        
        return y

    @staticmethod
    def f_prime(x):
        """ Applies the derivative of ReLU to x.
         
        
        Parameters
        ----------
        x: torch.Tensor (any shape, typically (batch_size, dimension))
            The input to the ReLU derivative
            
            
        Returns
        -------
        y: torch.Tensor(same shape as x)
            The result of applying the derivative of ReLU to x
        """
        
        y = (x.sign()+1)/2
        
        return y


class Tanh(Activation):
    """ Hyperbolic tangent class.
    Overrides f and f_prime.
    
    Methods
    -------
    f(x):
        Apply the Tanh function to x.
        
    f_prime(x):
        Apply the derivative of Tanh to x.
    """
    
    @staticmethod
    def f(x):
        """ Applies the Tanh function to x.
         
        
        Parameters
        ----------
        x: torch.Tensor (any shape, typically (batch_size, dimension))
            The input to the Tanh function
            
            
        Returns
        -------
        y: torch.Tensor(same shape as x)
            The result of applying Tanh to x
        """
        y = x.tanh()
        
        return y

    @staticmethod
    def f_prime(x):
        """ Applies the derivative of Tanh to x.
         
        
        Parameters
        ----------
        x: torch.Tensor (any shape, typically (batch_size, dimension))
            The input to the Tanh derivative
            
            
        Returns
        -------
        y: torch.Tensor(same shape as x)
            The result of applying the derivative of Tanh to x
        """
        
        y = 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
        
        return y

class LossMSE(Module):
    """ Mean Squared Error Loss. 
    """
    
    def forward(self , prediction, target):
        """ Computes the MSE of a prediction given a target.
        Stores the error for the backward pass. 
        
        Parameters
        ----------
        prediction: torch.Tensor of shape (batch_size, dimension)
            The predicted vectors
        target: torch.Tensor, same shape as prediction
            The target vectors
            
        Returns
        -------
        loss: torch.Tensor of shape []
            The sum of the MSE loss of each batch
        """
        
        error = target - prediction
        self.error = error
        self.d = prediction.shape[1]
        loss = error.pow(2).sum()/self.d

        return loss

    def backward(self):
        """ Computes the derivative of the MSE loss with respect to its input.
         
        Returns
        -------
        grad_input: torch.Tensor(same shape as self.error)
            The gradient with respect to the input of the loss function
        """
        
        return -2*self.error/self.d

class LossBCE(Module):
    """ Binary Cross Entropy Loss. 
    """
    
    def forward(self , prediction, target):
        """ Computes the BCE of a prediction given a target.
        Stores the prediction and target for the backward pass. 
        
        Parameters
        ----------
        prediction: torch.Tensor of shape (batch_size, 1)
            The predicted 1D vectors
        target: torch.Tensor, same shape as prediction
            The target 1D vectors, expected to be 0 or 1
            
        Returns
        -------
        loss: torch.Tensor of shape []
            The sum of the MSE loss of each batch
        """
        
        self.p = prediction
        self.t = target
        loss = (prediction.logaddexp(empty(prediction.shape).zero_()) - target*prediction).sum()/target.shape[1]

        return loss

    def backward(self):
        """ Computes the derivative of the MSE loss with respect to its input.
         
        Returns
        -------
        grad_input: torch.Tensor(same shape as self.error)
            The gradient with respect to the input of the loss function
        """
        
        exp = self.p.exp()
        grad_input = (exp/(1+exp) - self.t)/self.t.shape[1]

        return grad_input


class Linear(Module):
    """ Fully Connected Layer. 
    """
    
    def __init__(self, input_dim, output_dim, sigma = 1.):
        """ Constructor
        Intializes the weight matrix and bias vector, set their gradient to 0 and declares the layer as trainable.
        
        Parameters
        ----------
        input_dim: int
            The dimension of the input vectors
        output_dim: int
            The dimension of the output vectors
        sigma: float
            The standard deviation of the normal distribution to initialize weights and biases.
        """
        
        self.W = empty((output_dim, input_dim)).normal_(0, sigma)
        self.b = empty(output_dim).normal_(0, sigma)
        self.zero_grad()
        self.trainable = True

    def zero_grad(self):
        """ Re-set the gradients og the weights and biases to 0. """
        self.grad_W = empty(self.W.shape).zero_()
        self.grad_b = empty(self.b.shape).zero_()


    def forward(self, input_):
        """ Computes the forward pass.
        Stores the input_ for the backward pass. 
        
        Parameters
        ----------
        input_: torch.Tensor of shape (batch_size, input_dim)
            The input vectors
            
        Returns
        -------
        output: torch.Tensor of shape (batch_size, output_dim)
            The output vectors
        """
        
        self.input = input_
        output = (self.W.mm(input_.T) + self.b.unsqueeze(1)).T

        return output


    def backward(self, grad_output):
        """ Computes the backward pass.
        
        Parameters
        ----------
        grad_output: torch.Tensor of shape (batch_size, output_dim)
            The gradient at the output of the fully connected layer
            
        Returns
        -------
        grad_input: torch.Tensor of shape (batch_size, intput_dim)
            The gradient at the input of the fully connected layer
        """
        

        grad_input = grad_output.mm(self.W)
        self.grad_W += grad_output.T.mm(self.input)
        self.grad_b += grad_output.T.sum(axis = 1)

        return grad_input


    def param(self):
        """ Returns a list of 2 tuples: [(weights, weights gradient), (biases, biases gradient)]"""
        
        return [(self.W, self.grad_W), (self.b, self.grad_b)]

class Sequential(Module):
    """ Sequential neural network 
    """
    
    def __init__(self, *layers):
        """ Constructor.
        
        Parameters
        ----------
        *layers: Modules
            All Modules in the network, given in sequential order
        """
        
        self.layers = layers

    def forward(self , input_):
        """ Computes the forward pass through all layers.
        
        Parameters
        ----------
        input_: torch.Tensor of shape (batch_size, input_dim)
            The input to the network
            
        Returns
        -------
        output: torch.Tensor of shape (batch_size, output_dim)
            The output of the network
        """
        
        output = input_
        for layer in self.layers:
            output = layer.forward(output)
            
        return output

    def backward(self , grad_output):
        """ Computes the backward pass through all layers.
        
        Parameters
        ----------
        grad_output: torch.Tensor of shape (batch_size, output_dim)
            The gradient at the output of the network, before the loss function
            
        Returns
        -------
        grad_input: torch.Tensor of shape (batch_size, input_dim)
            The gradient at the input of the fully network
        """
        
        grad_input = grad_output
        for layer in reversed(self.layers):
            grad_input = layer.backward(grad_input)
            
        return grad_input

    def zero_grad(self):
        """ Re-set the gradient of all trainable layers to 0.  """
        for layer in self.layers:
            if layer.trainable:
                layer.zero_grad()

    def param(self):
        return []
