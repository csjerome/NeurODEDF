import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.integrate import solve_ivp
import shelve


class DampledPendulum(Dataset):
    __default_params = OrderedDict(omega0_square=(2 * np.pi / 6) ** 2, alpha=0.5)

    def __init__(self, path, num_seq, time_horizon, dt, params=None, group='train'):
        super().__init__()
        self.len = num_seq
        self.time_horizon = float(time_horizon)  # total time
        self.dt = float(dt)  # time step
        self.params = OrderedDict()
        if params is None:
            self.params.update(self.__default_params)
        else:
            self.params.update(params)
        self.group = group
        self.data = shelve.open(path)

    def _f(self, t, x,u):  # coords = [q,p]
        omega0_square, alpha = list(self.params.values())
        q, p = np.split(x, 2)
        dqdt = p
        dpdt = -omega0_square * np.sin(q) - alpha * p + u[int(t/self.dt)-1]
        return np.concatenate([dqdt, dpdt], axis=-1)

    def _get_initial_condition(self, seed):
        np.random.seed(seed if self.group == 'train' else np.iinfo(np.int32).max - seed)
        y0 = np.random.rand(2) * 2.0 - 1
        radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius
        return y0

    def _get_action(self,seed):
        s = 1
        np.random.seed(seed if self.group == 'train' else np.iinfo(np.int32).max - seed)
        u = np.random.normal(0,s,(int(self.time_horizon/self.dt),1))
        return u.astype(np.float32)

    def __getitem__(self, index):
        t_eval = torch.from_numpy(np.arange(0, self.time_horizon, self.dt))
        if self.data.get('states_' + str(index)) is None:
            y0 = self._get_initial_condition(index)
            u = self._get_action(index)
            states = solve_ivp(fun=self._f, t_span=(0, self.time_horizon), y0=y0, method='RK45', t_eval=t_eval, rtol=1e-10, args = (u,)).y

            self.data['states_' + str(index)] = states
            self.data['actions_' + str(index)] = u
            states = torch.from_numpy(states).float()
        else:
            states = torch.from_numpy(self.data['states_'+ str(index)]).float()
            u = torch.from_numpy(self.data['actions_'+ str(index)]).float()

        return {'states': states, 'actions': u, 't': t_eval.float()}

    def __len__(self):
        return self.len

class Pont_roulant(Dataset):
    __default_params = OrderedDict(M_mass = torch.tensor(25.), m_mass = torch.tensor(8.), length = torch.tensor(1.2))

    def __init__(self, path, num_seq, time_horizon, dt, params=None, group='train'):
        super().__init__()
        self.len = num_seq
        self.time_horizon = float(time_horizon)  # total time
        self.dt = float(dt)  # time step
        self.params = OrderedDict()
        if params is None:
            self.params.update(self.__default_params)
        else:
            self.params.update(params)
        self.group = group
        self.data = shelve.open(path)

    def _f(self, t, x,u):  # coords = [q,p]
        #omega0_square, alpha = list(self.params.values())

        th, dth, x, dx = np.split(x, 4)
        F = u[int(t/self.dt)-1]
        #print(th*180/np.pi)
        dt = self.dt
        M = self.params['M_mass'].item()
        m = self.params['m_mass'].item()
        l = self.params['length'].item()

        g = 9.81

        #dth, x, dx = dth/self.dt, x*l*10, dx/self.dt*l*10

        dTh = dth
        ddTh = -((M+m)*g*np.sin(th)+m*l*np.sin(th)*np.cos(th)*dth**2+np.cos(th)*F)/(M+m*np.sin(th)**2)/l
        dX = dx
        ddX = (m*l*dth**2*np.sin(th)+m*g*np.sin(th)*np.cos(th)+ F)/(M+m*np.sin(th)**2)

        #dTh, ddTh, dX, ddX = dTh*dt, ddTh*dt**2, dX/l*dt/10, ddX*dt**2/l/10
        return np.concatenate([dTh,ddTh,dX,ddX], axis=-1)

    def _get_initial_condition(self, seed):
        np.random.seed(seed if self.group == 'train' else np.iinfo(np.int32).max - seed)
        y0 = np.random.rand(4) * 2.0 - 1
        radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius
        y0 = np.zeros(4)
        return y0

    def _get_action(self,seed):
        s = 0.1
        k = 1
        np.random.seed(seed if self.group == 'train' else np.iinfo(np.int32).max - seed)
        #u = np.random.normal(0,s,(int(self.time_horizon/self.dt),1))*A
        noise = np.random.normal(0,np.sqrt(self.dt))
        u = np.zeros((int(self.time_horizon/self.dt),1))
        u[0,0] = noise*s
        for i in range(len(u)-1):
            noise = np.random.normal(0,np.sqrt(self.dt))
            u[i+1,0] = u[i,0]*(1- k*self.dt) + s*noise
        return u.astype(np.float32)

    def _get_action_sbpa(self,seed):
        A = 10 #amplitude maximal de la commande
        N = 2
        T = int(self.time_horizon/N/self.dt)
        np.random.seed(seed if self.group == 'train' else np.iinfo(np.int32).max - seed)
        u = np.zeros((int(self.time_horizon/self.dt),1))
        for i in range(N):
            noise = (np.random.rand()*2-1)*A
            u[i*T:(i+1)*T,0] = np.ones(T)*noise
        u[T*N:,0] = noise*np.ones(len(u[T*N:,0]))
        return u.astype(np.float32)

    def __getitem__(self, index):
        t_eval = torch.from_numpy(np.arange(0, self.time_horizon, self.dt))
        if self.data.get('states_' + str(index)) is None:
            y0 = self._get_initial_condition(index)
            u = self._get_action_sbpa(index)
            states = solve_ivp(fun=self._f, t_span=(0, self.time_horizon), y0=y0, method='RK23', t_eval=t_eval, rtol=1e-10, args = (u,),max_step=10).y
            self.data['states_' + str(index)] = states
            self.data['actions_' + str(index)] = u
            states = torch.from_numpy(states).float()
            u = torch.from_numpy(u).float()
        else:
            states = torch.from_numpy(self.data['states_'+ str(index)]).float()
            u = torch.from_numpy(self.data['actions_'+ str(index)]).float()
        return {'states': states, 'actions': u, 't': t_eval.float()}

    def __len__(self):
        return self.len



def create_dataset(batch_size=25, dt = 0.5, time_horizon = 20):
    dataset_train_params = {
        'num_seq': 25,
        'time_horizon': time_horizon,
        'dt': dt,
        'group': 'train',
        'path': '.\exp\_train_pont_adim',
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params['num_seq'] = 25
    dataset_test_params['group'] = 'test'
    dataset_test_params['path'] = '.\exp\_test_pont_adim'

    #dataset_train = DampledPendulum(**dataset_train_params)
    #dataset_test  = DampledPendulum(**dataset_test_params)

    dataset_train = Pont_roulant(**dataset_train_params)
    dataset_test  = Pont_roulant(**dataset_test_params)


    dataloader_train_params = {
        'dataset'    : dataset_train,
        'batch_size' : batch_size,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : True,
    }

    dataloader_test_params = {
        'dataset'    : dataset_test,
        'batch_size' : batch_size,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : False,
    }
    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test  = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test