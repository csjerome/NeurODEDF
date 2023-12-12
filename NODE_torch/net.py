import torch
from torchdiffeq import odeint
import torch.nn as nn
from collections import OrderedDict

class DampedPendulumParamPDE(nn.Module):
    def __init__(self, is_complete=False, real_params=None):
        super().__init__()
        self.real_params = real_params
        self.is_complete = is_complete
        self.params_org = nn.ParameterDict({
            'omega0_square_org': nn.Parameter(torch.tensor(0.2)),
            'alpha_org': nn.Parameter(torch.tensor(0.1)),
        })
        self.params = OrderedDict()
        if real_params is not None:
            self.params.update(real_params)

    def forward(self, state):
        if self.real_params is None:
            self.params['omega0_square'] = self.params_org['omega0_square_org'].item()

        q = state[:,0:1]
        p = state[:,1:2]

        if self.is_complete:
            if self.real_params is None:
                self.params['alpha'] = self.params_org['alpha_org'].item()
            (omega0_square, alpha) = list(self.params.values())
            dqdt = p
            dpdt = - omega0_square * torch.sin(q) - alpha * p
        else:
            (omega0_square, ) = list(self.params.values())
            dqdt = p
            dpdt = - omega0_square * torch.sin(q)

        return torch.cat([dqdt, dpdt], dim=1)
class MLP(nn.Module):
    def __init__(self, state_c, hidden):
        super().__init__()
        self.state_c = state_c
        self.net = nn.Sequential(
            nn.Linear(state_c, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, state_c),
        )
        self.net.apply(self.init_fc)


    def init_fc(self, m):
        if hasattr(m, 'weight'):
            nn.init.orthogonal_(m.weight.data, gain=0.2)
    def forward(self, x):
        return self.net(x)

    def get_derivatives(self, x):
        batch_size, nc, T = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(batch_size * T, nc)
        x = self.forward(x)
        x = x.view(batch_size, T, self.state_c)
        x = x.permute(0, 2, 1).contiguous()
        return x

class DerivativeEstimator(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        self.is_augmented = is_augmented


    def forward(self, t, state):
        res_phy = self.model_phy(state)
        if self.is_augmented:
            res_aug = self.model_aug(state)
            return res_phy + res_aug
        else:
            return res_phy

class Forecaster(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented, method='rk4', options=None):
        super().__init__()

        self.model_phy = model_phy
        self.model_aug = model_aug

        self.derivative_estimator = DerivativeEstimator(self.model_phy, self.model_aug, is_augmented=is_augmented)
        self.method = method
        self.options = options
        self.int_ = odeint

    def forward(self, y0, t):
        # y0 = y[:,:,0]
        res = self.int_(self.derivative_estimator, y0=y0, t=t, method=self.method, options=self.options)
        # res: T x batch_size x n_c (x h x w)
        dim_seq = y0.dim() + 1
        dims = [1, 2, 0] + list(range(dim_seq))[3:]
        return res.permute(*dims)   # batch_size x n_c x T (x h x w)

    def get_pde_params(self):
        return self.model_phy.params
#%%
