from linear import Linear
from act_func import Tanh, Softmax, Sigmoid
import numpy as np

class LSTM:
    def __init__(self, hidden_size, vocab_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.z_size = vocab_size + hidden_size

        self.Wg = Linear(self.z_size, hidden_size, bias = True)
        self.Wi = Linear(self.z_size, hidden_size, bias = True)
        self.Wf = Linear(self.z_size, hidden_size, bias = True)
        self.Wo = Linear(self.z_size, hidden_size, bias = True)
        self.Wv = Linear(hidden_size, vocab_size, bias = True)

        self.params = self.Wg, self.Wi, self.Wf, self.Wo, self.Wv

        self.tanh = Tanh()
        self.softmax = Softmax()
        self.sigmoid = Sigmoid()

    def __call__(self, x, h_prev, C_prev):
        """
        Arguments:
        x -- your input data at timestep "t", numpy array of shape (n_x, m).
        h_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        C_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)

        Returns:
        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s -- lists of size m containing the computations in each forward pass
        outputs -- prediction at timestep "t", numpy array of shape (n_v, m)
        """
        assert h_prev.shape == (self.hidden_size , 1)
        assert C_prev.shape == (self.hidden_size , 1)

        self.h_s, self.C_s, self.i_s, self.f_s, self.o_s, self.g_s, self.z_s, outputs = [h_prev], [C_prev], [], [], [], [], [], []

        for word in x:
            z = np.row_stack((h_prev, word))
            f = self.sigmoid(self.Wf(z))
            i = self.sigmoid(self.Wi(z))
            g = self.tanh(self.Wg(z))
            C_prev = f*C_prev + i*g
            o = self.sigmoid(self.Wo(z))
            h_prev = o * self.tanh(C_prev)
            output = self.softmax(self.Wv(h_prev))

            outputs.append(output)
            self.z_s.append(z)
            self.i_s.append(i)
            self.f_s.append(f)
            self.o_s.append(o)
            self.g_s.append(g)
            self.h_s.append(h_prev)
            self.C_s.append(C_prev)

        return outputs

    def backward(self, outputs, targets, do_grad = False):
        loss = 0

        for t in reversed(range(len(outputs))):
            loss += -np.mean(np.log(outputs[t]+1e-12) * targets[t])

            d_v = outputs[t].copy()
            d_v[np.argmax(targets[t])] -= 1

            if do_grad:
                o1 = self.softmax.backward(d_v)
                o2 = self.Wv.backward(o1, self.h_s[t+1])

                o3 = self.tanh(self.C_s[t])*o2
                o4 = self.sigmoid.backward(o3, self.o_s[t])
                self.Wo.backward(o4, self.z_s[t])

                o6 = self.tanh.backward(o2,self.tanh(self.C_s[t]))
                o7 = self.tanh.backward(o6*self.i_s[t],self.g_s[t])
                self.Wg.backward(o7, self.z_s[t])

                o9 = self.sigmoid.backward(o6*self.g_s[t],self.i_s[t])
                self.Wi.backward(o9, self.z_s[t])

                o11 = self.sigmoid.backward(o6*self.C_s[t-1],self.f_s[t])
                self.Wf.backward(o11,self.z_s[t])

        self.h_s, self.C_s, self.i_s, self.f_s, self.o_s, self.g_s, self.z_s, outputs = [], [], [], [], [], [], [], []

        return loss

    def step(self, lr):
        for param in self.params:
            param.step(lr)
