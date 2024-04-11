import torch
from torchdiffeq import odeint_adjoint as odeint
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class DampedPendulumParamPDE(nn.Module):
    def __init__(self, is_complete=False, real_params=None):
        super().__init__()
        self.real_params = real_params
        self.is_complete = is_complete
        self.params_org = nn.ParameterDict({
            'omega0_square_org': nn.Parameter(torch.tensor(0.9), requires_grad = True),
            'alpha_org': nn.Parameter(torch.tensor(0.3), requires_grad= True),
        })
        self.params = OrderedDict()
        if real_params is not None:
            self.params.update(real_params)

    def forward(self, state):
        if self.real_params is None:
            self.params['omega0_square'] = self.params_org['omega0_square_org']

        q = state[:,0:1]
        p = state[:,1:2]

        if self.is_complete:
            if self.real_params is None:
                self.params['alpha'] = self.params_org['alpha_org']
            (omega0_square, alpha) = list(self.params.values())
            dqdt = p
            dpdt = - omega0_square * torch.sin(q) - alpha * p
        else:
            (omega0_square, ) = list(self.params.values())
            dqdt = p
            dpdt = - omega0_square * torch.sin(q)

        return torch.cat([dqdt, dpdt], dim=1)


class Pont_roulantPDE(nn.Module):
    def __init__(self,dt, is_complete=False, real_params=None):
        super().__init__()
        self.dt = dt
        self.real_params = real_params
        self.is_complete = is_complete
        self.params_org = nn.ParameterDict({
            'M_mass': nn.Parameter(torch.tensor(23.), requires_grad = True),
            'm_mass': nn.Parameter(torch.tensor(8.5), requires_grad = True),
            'length': nn.Parameter(torch.tensor(1.0), requires_grad= True),
        })
        self.params = OrderedDict()
        if real_params is not None:
            self.params.update(real_params)

    def forward(self, state, F):
        if self.real_params is None:
            self.params['M_mass'] = self.params_org['M_mass']
            self.params['m_mass'] = self.params_org['m_mass']
            self.params['length'] = self.params_org['length']
        '''
        q = state[:,0:1]
        p = state[:,1:2]
        '''
        th, dth, x, dx = torch.tensor_split(state, 4, dim = 1)
        g = 9.81
        (M, m, l) = list(self.params.values())
        dt = self.dt

        #dth, x, dx = dth/dt, x*l*10, dx/dt*l*10

        if self.is_complete:
            dTh = dth
            ddTh = -((M+m)*g*torch.sin(th)+m*l*torch.sin(th)*torch.cos(th)*dth**2+torch.cos(th)*F)/(M+m*torch.sin(th)**2)/l
            dX = dx
            ddX = (m*l*dth**2*torch.sin(th)+m*g*torch.sin(th)*torch.cos(th)+ F)/(M+m*torch.sin(th)**2)

        else:
            dTh = dth
            ddTh = -((M+m)*g*torch.sin(th)+m*l*torch.sin(th)*torch.cos(th)*dth**2)/(M+m*torch.sin(th)**2)/l
            dX = dx
            ddX = (m*l*dth**2*torch.sin(th)+m*g*torch.sin(th)*torch.cos(th))/(M+m*torch.sin(th)**2)

        #dTh, ddTh, dX, ddX = dTh*dt, ddTh*dt**2, dX/l*dt/10, ddX*dt**2/l/10
        return torch.concatenate([dTh,ddTh,dX,ddX], axis=-1)
class MLP(nn.Module):
    def __init__(self, state_c, hidden, input = None):
        if input is None: input = 0
        super().__init__()
        self.state_c = state_c
        self.net = nn.Sequential(
            nn.Linear(state_c + input, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, state_c),
        )
        self.net.apply(self.init_fc)


    def init_fc(self, m):
        if hasattr(m, 'weight'):
            nn.init.orthogonal_(m.weight.data, gain=0.2)
    def forward(self, x, u = None):
        return self.net(torch.cat((x,u),1))

    def get_derivatives(self, x,u = None):
        batch_size, nc, T = x.shape

        _, _, nu = u.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(batch_size * T, nc)
        u = u.view(batch_size * T, nu)
        x = self.forward(x,u)
        x = x.view(batch_size, T, self.state_c)
        x = x.permute(0, 2, 1).contiguous()
        return x

class DerivativeEstimator(nn.Module):
    def __init__(self, model_phy, model_aug):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        self.action = None


    def forward(self, t, state):
        res = 0
        if self.model_phy != None:
            res += self.model_phy(state,self.action[:,int(t/self.dt)-1])
        if self.model_aug != None:
            res += self.model_aug(state,self.action[:,int(t/self.dt)-1])
        return res

class Forecaster(nn.Module):
    def __init__(self, model_phy, model_aug = None, method='dopri5'):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug

        self.derivative_estimator = DerivativeEstimator(self.model_phy, self.model_aug)
        self.method = method

        self.int_ = odeint

    def forward(self, y0, t, u = None):
        # y0 = y[:,:,0]
        self.derivative_estimator.action = u
        self.derivative_estimator.dt = t[1]-t[0]
        res = self.int_(self.derivative_estimator, y0=y0, t=t, method=self.method)
        # res: T x batch_size x n_c (x h x w)
        dim_seq = y0.dim() + 1
        dims = [1, 2, 0] + list(range(dim_seq))[3:]
        return res.permute(*dims)   # batch_size x n_c x T (x h x w)

#%%
