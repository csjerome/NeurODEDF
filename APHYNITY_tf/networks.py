
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
import numpy as np

from collections import OrderedDict

import tensorflow.keras.layers as kr
import tensorflow as tf

class DampedPendulumParamPDE(kr.Layer):
    def __init__(self, is_complete=False, real_params=None):
        super().__init__()
        self.real_params = real_params
        self.is_complete = is_complete
        self.params_org = {
            'omega0_square_org': tf.Variable(0.2),
            'alpha_org': tf.Variable(0.1)}
        self.params = OrderedDict()
        if real_params is not None:
            self.params.update(real_params)

    def call(self, state):
        if self.real_params is None:
            self.params['omega0_square'] = self.params_org['omega0_square_org']

        q = state[:,0:1]
        p = state[:,1:2]
        
        if self.is_complete:
            if self.real_params is None:
                self.params['alpha'] = self.params_org['alpha_org']
            (omega0_square, alpha) = list(self.params.values())
            dqdt = p
            dpdt = - omega0_square * tf.sin(q) - alpha * p
        else:
            (omega0_square, ) = list(self.params.values())
            dqdt = p
            dpdt = - omega0_square * tf.sin(q)

        return tf.concat([dqdt, dpdt], axis=1)


class MLP(kr.Layer):
    def __init__(self, state_c, hidden):
        super().__init__()
        self.state_c = state_c
        self.net = tf.keras.models.Sequential()
        self.net.add(kr.Dense(hidden, input_shape =(state_c,), activation = 'relu'))
        self.net.add(kr.Dense(hidden, activation= 'relu'))
        self.net.add(kr.Dense(state_c, input_shape = (hidden,)))
    
    def call(self, x):
        return self.net(x)

    def get_derivatives(self, x):
        batch_size, nc, T = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(batch_size * T, nc)
        x = self.forward(x)
        x = x.view(batch_size, T, self.state_c)
        x = x.permute(0, 2, 1).contiguous()
        return x
