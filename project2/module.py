class ReLU():
    @staticmethod
    def f(x):
        return x.clamp(min = 0)

    @staticmethod
    def f_prime(x):
        return (x.sign()+1)/2

    def forward(self, input_):
        output = self.f(input_)
        self.input = input_

        return output

    def backward(self, grad_output):
        input_ = self.input
        grad_input = grad_output *  self.f_prime(input_)

        return grad_input
