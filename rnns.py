from linear import Linear
from act_func import Tanh, Softmax
import numpy as np

class RNN_completed:
    def __init__(self, hidden_size, vocab_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.U = Linear(vocab_size, hidden_size, bias=False)
        self.V = Linear(hidden_size, hidden_size, bias=True)
        self.W = Linear(hidden_size, vocab_size, bias=True)

        self.params = self.W, self.V, self.U

        self.tanh = Tanh()
        self.softmax = Softmax()

    def __call__(self,x, hidden_state=None):

        self.seq_len = x.shape[0]

        if hidden_state is None:
            hidden_state = np.zeros((self.hidden_size,1))

        self.hidden_states = [hidden_state]
        outputs = []
        self.x = x

        for word in x:
            a1 = self.U(word)
            a2 = self.V(hidden_state)
            a3 = a1 + a2
            hidden_state = self.tanh(a3)
            a4 = self.W(hidden_state)
            out = self.softmax(a4)

            outputs.append(out)
            self.hidden_states.append(hidden_state)
        return  np.array(outputs), self.hidden_states


    def backward(self, outputs, targets, do_grad = False):
        loss = 0

        for t in reversed(range(self.seq_len)):
            loss += -np.mean(np.log(outputs[t]+1e-12) * targets[t])

            d_o = outputs[t].copy()
            d_o[np.argmax(targets[t])] -= 1

            if do_grad:
                o1 = self.softmax.backward(d_o)
                o2 = self.W.backward(o1, self.hidden_states[t+1])
                o3 = self.tanh.backward(o2, self.hidden_states[t+1])
                o4 = self.V.backward(o3, self.hidden_states[t])
                o5 = self.U.backward(o3, self.x[t])
        self.hidden_states = []
        return loss

    def step(self, lr):
        for param in self.params:
            param.step(lr)


class RNN_stupid:
    def __init__(self, hidden_size, vocab_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.U = self.init_orthogonal(np.random.randn(hidden_size, vocab_size))
        self.V = self.init_orthogonal(np.random.randn(hidden_size, hidden_size))
        self.bv = np.random.randn(hidden_size, 1)
        self.W = self.init_orthogonal(np.random.randn(vocab_size, hidden_size))
        self.bw = np.random.randn(vocab_size, 1)

        self.tanh = Tanh()
        self.softmax = Softmax()

    def __call__(self, x, hidden_state=None):

        if hidden_state is None:
            hidden_state = np.zeros((self.hidden_size, 1))

        self.x = x

        outputs = []
        self.hidden_states = [hidden_state]

        for t in range(len(x)):
            o1 = np.dot(self.U,x[t])
            o2 = np.dot(self.V, hidden_state) + self.bv
            hidden_state = self.tanh(o1+o2)
            o4 = np.dot(self.W, hidden_state) + self.bw
            out = self.softmax(o4)
            outputs.append(out)
            self.hidden_states.append(hidden_state)

        self.hidden_states = np.array(self.hidden_states)

        return np.array(outputs), self.hidden_states

    def backward(self, outputs, targets, do_grad = True):
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        dV = np.zeros_like(self.V)
        dbv = np.zeros_like(self.bv)
        dbw = np.zeros_like(self.bw)
        loss = 0

        for t in reversed(range(len(outputs))):
            loss += -np.mean(np.log(outputs[t]+1e-12) * targets[t])
            if do_grad:
                d_o = outputs[t].copy()
                d_o[np.argmax(targets[t])] -= 1

                dW += np.dot(d_o, self.hidden_states[t+1].T)
                dbw += d_o

                dh = np.dot(self.W.T, d_o)
                df = (1 - self.tanh(self.hidden_states[t+1])**2)*dh

                dV +=  np.dot(df, self.hidden_states[t].T)
                dbv += df

                dU += np.dot(df, self.x[t].T)

        if do_grad:
            self.grads = [dW, dbw, dV, dbv, dU]
            self.clip_gradient_norm()
        return loss

    def step(self, lr):
        self.W -= lr*self.grads[0]
        self.bw -= lr*self.grads[1]
        self.V -= lr*self.grads[2]
        self.bv -= lr*self.grads[3]
        self.U -= lr*self.grads[4]

    def clip_gradient_norm(self, max_norm=0.25):
        """
        Clips gradients to have a maximum norm of `max_norm`.
        This is to prevent the exploding gradients problem.
        """
        # Set the maximum of the norm to be of type float
        max_norm = float(max_norm)
        total_norm = 0

        # Calculate the L2 norm squared for each gradient and add them to the total norm
        for grad in self.grads:
            grad_norm = np.sum(np.power(grad, 2))
            total_norm += grad_norm

        total_norm = np.sqrt(total_norm)

        # Calculate clipping coeficient
        clip_coef = max_norm / (total_norm + 1e-6)

        # If the total norm is larger than the maximum allowable norm, then clip the gradient
        if clip_coef < 1:
            for idx in range(len(self.grads)):
                self.grads[idx] *= clip_coef

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
