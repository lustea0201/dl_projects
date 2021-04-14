import unittest
from torch import tensor, equal, set_grad_enabled
from module import ReLU

set_grad_enabled(False) # REQUIRED

class TestModule(unittest.TestCase):

    def test_ReLU(self):
        """ Test ReLU independently of other modules.
        Scenario:
        x0 -ReLU-> x1 -> MSE(x1, t) = y

        forward:
        - x0 = [-10, 1.5, -1, 4] (fixed)
        - t = [2, 2, 2, 2] (fixed)
        - x1 expected to be [0, 1.5, 0, 4]
        - y should be 3.0625 in that case (explicit)

        backward:
        y = 1/4*((x_11-2)**2 + (x_12-2)**2 + (x_13-2)**2 + (x_14-2)**2)
        thus:
        dy/dx1 = [(x11-2)/2, (x12-2)/2, (x13-2)/2, (x14-2)/2]
        - grad_output = [-0.5, 0.25, -0.5, 1.5] (fixed)
        then:
        - dy/dx0 = dy/dx1 * dx1/dx0 = grad_output * f_prime(x0)
        expected to be [0, 0.25, 0, 1.5]
        """

        relu = ReLU()
        x0 = tensor([-10., 1.5, -1., 4.])
        x1 = relu.forward(x0)
        self.assertTrue(equal(x1, tensor([0, 1.5, 0, 4])), 'ReLU forward not working')

        grad_output = tensor([-0.5, 0.25, -0.5, 1.5])
        grad_input = relu.backward(grad_output)
        self.assertTrue(equal(grad_input, tensor([0, 0.25, 0, 1.5])), 'ReLU backward not working')


if __name__ == '__main__':
    unittest.main()
