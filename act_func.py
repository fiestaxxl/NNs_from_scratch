import numpy as np

class Tanh:
    def __init__(self):
        # the layer has no parameters
        pass

    def tanh(self, x):
        x_safe = x + 1e-12
        f = np.tanh(x_safe)
        return f

    def __call__(self, x):
        # keep o for backward computation
        o = self.tanh(x)
        return o


    def backward(self, output_grad, x):
        """
        Calculate and return the gradient of the loss w.r.t. the input
        of tanh (given input x and the gradient
        w.r.t output of logistic non-linearity).

        :param x: np.array, input tensor for logistic non-linearity;
        :param output_grad: np.array, grad tensor w.r.t output of logistic non-linearity;
        :return: np.array, grad w.r.t input of logistic non-linearity

        """

        input_grad = (1-self.tanh(x)**2)*output_grad

        return input_grad


class Sigmoid:
    def __init__(self):
        # the layer has no parameters
        pass

    def sigmoid(self, x):
        x_safe = x + 1e-12
        f = 1 / (1 + np.exp(-x_safe))
        return f

    def __call__(self, x):
        # keep o for backward computation
        o = self.sigmoid(x)
        return o


    def backward(self, output_grad, x):
        """
        Calculate and return the gradient of the loss w.r.t. the input
        of tanh (given input x and the gradient
        w.r.t output of logistic non-linearity).

        :param x: np.array, input tensor for logistic non-linearity;
        :param output_grad: np.array, grad tensor w.r.t output of logistic non-linearity;
        :return: np.array, grad w.r.t input of logistic non-linearity

        """

        input_grad = (self.sigmoid(x)*(1-self.sigmoid(x)))*output_grad


        return input_grad


class Softmax:
    def __init__(self):
        # the layer has no parameters
        pass

    def softmax(self, x):
        self.size = x.shape[0]
        x_safe = x + 1e-12
        f = np.exp(x_safe)/ np.sum(np.exp(x_safe))
        return f

    def __call__(self, x):
        # keep o for backward computation
        o = self.softmax(x)
        return o

    def backward(self, output_grad):
        return output_grad