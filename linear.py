import numpy as np

class Linear:
    def __init__(self, input_size, output_size, bias=True):
        # Trainable parameters of the layer and their gradients
        self.thetas = self.init_orthogonal(np.random.randn(output_size, input_size)) # the weight matrix of the layer (W)
        self.thetas_grads = np.zeros_like(self.thetas, dtype=np.float32) # gradient w.r.t. the weight matrix of the layer
        self.bias_cond = bias

        if bias:
            self.bias = np.random.randn(output_size, 1) # bias terms of the layer (b)
            self.bias_grads = np.zeros_like(self.bias, dtype=np.float32) # gradient w.r.t. bias terms of the linear layer
        else:
            self.bias = None # bias terms of the layer (b)
            self.bias_grads = None # gradient w.r.t. bias terms of the linear layer

    def __call__(self, x):
        if self.bias_cond:
            output = np.dot(self.thetas, x) + self.bias
        else:
            output = np.dot(self.thetas, x)
        return output

    def backward(self, output_grad, x):
        """
        Calculate and return gradient of the loss w.r.t. the input of linear layer given the input x and the gradient
        w.r.t output of linear layer. You should also calculate and update gradients of layer parameters.
        :param x: np.array, input tensor for linear layer;
        :param output_grad: np.array, grad tensor w.r.t output of linear layer;
        :return: np.array, grad w.r.t input of linear layer
        """

        input_grad = np.dot(self.thetas.T, output_grad)
        self.thetas_grads += np.dot(output_grad, x.T)

        if self.bias_cond:
            self.bias_grads += output_grad

        return input_grad

    def step(self, learning_rate):
        #self._clip()
        self.thetas -= self.thetas_grads * learning_rate
        self.thetas_grads = np.zeros_like(self.thetas_grads, dtype=np.float32)

        if self.bias_cond:
            self.bias -= self.bias_grads * learning_rate
            self.bias_grads = np.zeros_like(self.bias_grads, dtype=np.float32)




    def _clip(self, clip_value=0.95):
        for gradient in [self.thetas_grads, self.bias_grads]:
                if gradient is not None:
                    np.clip(gradient, -clip_value, clip_value, out=gradient)


    def init_orthogonal(self, param):

        """
        Initializes weight parameters orthogonally.

        Refer to this paper for an explanation of this initialization:
        https://arxiv.org/abs/1312.6120
        """

        if param.ndim < 2:
            raise ValueError("Only parameters with 2 or more dimensions are supported.")

        rows, cols = param.shape

        new_param = np.random.randn(rows, cols)

        if rows < cols:
            new_param = new_param.T

        # Compute QR factorization
        q, r = np.linalg.qr(new_param)

        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = np.diag(r, 0)
        ph = np.sign(d)
        q *= ph

        if rows < cols:
            q = q.T

        new_param = q

        return new_param
