import torch
from torchdiffeq import odeint_adjoint as odeint
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from net import *
import matplotlib as mpl
mpl.use('TkAgg')
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import shutil

class FMU:
    def __init__(self, fmu_filename, inputs, outputs):
        self.filename = fmu_filename
        self.model_description = read_model_description(fmu_filename, validate=False)

        # Valeurs de référence
        self.vrs = {}
        for variable in self.model_description.modelVariables:
            self.vrs[variable.name] = variable.valueReference

        # Entrée et sortie
        self.vrs_input = []
        for i in inputs:
            self.vrs_input.append(self.vrs[i])
        self.vrs_output = []
        for o in outputs:
            self.vrs_output.append(self.vrs[o])

        # Extraire la fmu
        self.unzipdir = extract(fmu_filename)

        self.fmu = FMU2Slave(guid=self.model_description.guid,
                             unzipDirectory=self.unzipdir,
                             modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                             instanceName='instance')

        # Initialisation de now_input à None
        self.now_input = None

    def initialize(self, start_time):
        # Initialisation
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=start_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.passed_in = np.array([])
        self.passed_out = np.array([])

    def setInputs(self, inputs):
        self.fmu.setReal(self.vrs_input, inputs)
        self.now_input = inputs 

    def getOutputs(self):
        return self.fmu.getReal(self.vrs_output)

    def doStep(self, time, step_size):
        self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
        self.passed_in = np.append(self.passed_in, self.now_input)
        outputs = self.getOutputs()
        self.passed_out = np.append(self.passed_out, outputs)
        return outputs

    def terminate(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        
        shutil.rmtree(self.unzipdir)

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
        th, dth, x, dx = torch.tensor_split(state, 4, dim = 1)
        g = 9.81
        (M, m, l) = list(self.params.values())
        dt = self.dt

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

# conditions initiales et trajectoire de référence
condinit = np.array([0.3, 0.,0.,0.])
Xreference = np.ones(1000)
# paramètres du pendule
omega0_square = 1
alpha = 0.1
sat = 2
# paramètres du temps
Tf, dt = 30, 0.01
# paramètres de la cross entropy
Nsignaux, Horizon, Nkeptsignals, Nstepcrossentropy = 50, 15, 5, 5
# paramètres de pondération du coût
q, r = 10, 1

file = 'model_1.26e+00.pt'
param = torch.load('./exp/'+ file)
dt, horizon = param['dt'], param['horizon']
model_phy = param['model_phy']
model_aug = param['model_aug']
net = Forecaster(model_phy=model_phy, model_aug=model_aug)
net.load_state_dict(param['model_state_dict'])
net.eval()

# Charger le FMU
fmu = FMU('Pont_roulant.fmu', inputs=['F'], outputs=['theta', 'omega', 'x', 'v'])
fmu.initialize(start_time=0.0)

def generatesignals(n, horizon, listemu, listesigma):
    L = [[0 for i in range(horizon)] for j in range(n)]
    for i in range(n):
        for j in range(horizon):
            L[i][j] = [rd.normalvariate(listemu[j], listesigma[j])]
    return np.array(L).astype(np.float32)

def cost(xexpected, xreference, u):
    Lp = torch.norm(xexpected - torch.tensor(xreference))
    return Lp

def keepbest(signauxtest, nkeptsignals, indexlist):
    L = [signauxtest[indexlist[i]] for i in range(nkeptsignals)]
    return L

def modelependulenn(x, u):
    x = np.array(x)
    u = torch.tensor(u)
    print(x,u)
    Y = torch.cat((x,u),-1)
    dy = net.forward(x,t,u)
    y = [dy[0], dy[1]] 
    return y

def standarddeviation(L):
    a = 0
    b = average(L)
    for i in range(len(L)):
        a = a + (L[i] - b)**2
    return np.sqrt(a)

def verifordre(L):
    for i in range (1, len(L)):
        if L[i] < L[i-1]:
            return False
    return True

# Initialisation de la commande précédente
previous_command = 0.0
scoredeb, scorefin =[], []
Statevector = condinit.astype(np.float32)
theta, omega, x, v, commande = [], [], [], [], []
for i in range(int(Tf/dt)):
    print(str(i)+'/'+str(int(Tf/dt)))
    mu = [0] * Horizon
    sigma = [1] * Horizon
    Y0 = np.repeat(Statevector[np.newaxis, :], Nsignaux, axis = 0)
    for l in range(Nstepcrossentropy):
        if l != 0 :
            Testsignals = generatesignals(Nsignaux, Horizon, mu, sigma)
        else :
            Testsignals = generatesignals(Nsignaux, Horizon, mu, sigma)
        for j in range(len(Testsignals)):
            for k in range(len(Testsignals[j])):
                for m in range(len(Testsignals[j][k])):
                    if Testsignals[j][k][m] > sat :
                        Testsignals[j][k][m] = sat
                    elif Testsignals[j][k][m] < -sat :
                        Testsignals[j][k][m] = -sat
        t = torch.arange(0,Horizon*dt,dt)
        Testresult = net.forward(torch.tensor(Y0), t,torch.tensor(Testsignals))
    
        Costlist = torch.zeros(Nsignaux)
        for j in range(Nsignaux):
            Xexpected = Testresult[j,0,:]
            Costlist[j]= cost(Xexpected, Xreference[i:i + Horizon], Testsignals[j])
        Indexlist = sorted(enumerate(Costlist), key = lambda  x : x[1])
        Indexlist = [j[0] for j in Indexlist]
        Bettersignals = Testsignals[Indexlist[:Nkeptsignals]]
        if not(verifordre([Costlist[Indexlist[i]] for i in range(len(Indexlist))])):
            print('ERREUR ERREUR ERREUR')
        mu, sigma = [], []
        if l == 0 :
            scoredeb.append(torch.mean(Costlist))
        if l == Nstepcrossentropy-1 :
            scorefin.append(torch.mean(Costlist))
        for j in range(Horizon):
            mu.append(np.mean([Bettersignals[k][j] for k in range(Nkeptsignals)]))
            sigma.append(np.std([Bettersignals[k][j] for k in range(Nkeptsignals)])) 
    net.derivative_estimator.action = torch.tensor([Bettersignals[0]])
    Statevector = net.int_(net.derivative_estimator, torch.tensor([Statevector]), t = torch.tensor([0.,dt])).detach().numpy()[1,0]
    
    # Filtre passe-bas pour lisser la commande
    alpha = 0.5 
    filtered_command = alpha * previous_command + (1 - alpha) * Bettersignals[0][0]
    previous_command = filtered_command  
    
    theta.append(Statevector[0])
    omega.append(Statevector[1])
    x.append(Statevector[2])
    v.append(Statevector[3])
    commande.append(filtered_command)

time = [i*dt for i in range(int(Tf/dt))]
timec = [i*dt for i in range(int(Tf/dt))]
consigne = [Xreference[i] for i in range(len(time))]
plt.plot(time, theta, color='blue', label = 'Angle du pendule')
plt.plot(time, x, color='red', label = 'Position')
plt.plot(timec, commande, color ='green', label = 'Commande')
plt.plot(time, consigne, color='black', label = 'Consigne')
plt.legend()
plt.show()