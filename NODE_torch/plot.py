import matplotlib.pyplot as plt
import torch
import numpy as np
from dataset import create_dataset
from net import *
from train import *
import shelve
import matplotlib as mpl
mpl.use('TkAgg')

file = 'model_9.27e-01.pt'
param = torch.load('./exp/'+ file)
train, test = create_dataset()

model_phy_option = 'data_driven'
model_aug_option = True

if model_phy_option == 'incomplete':
    model_phy = DampedPendulumParamPDE(is_complete=False, real_params=None)
elif model_phy_option == 'complete':
    model_phy = DampedPendulumParamPDE(is_complete=True, real_params=None)
elif model_phy_option == 'true':
    model_phy = DampedPendulumParamPDE(is_complete=True, real_params=train.dataset.params)
elif model_phy_option == 'data_driven':
    model_phy = None

if model_aug_option == True :
    model_aug = MLP(state_c=2, hidden=100,input=1)
else : model_aug = None

net = Forecaster(model_phy=model_phy, model_aug=model_aug)

net.load_state_dict(param['model_state_dict'])
net.eval()

#data = shelve.open('./_test.bak')
data = next(iter(test))
Y = data['states'][0]
t = data['t'][0]
u = data['actions'][0]
y0 = torch.unsqueeze(Y[:, 0],0)
u = torch.unsqueeze(u,0)
pred = net(y0,t,u)

for y in Y :
    plt.plot(t,y)

for y in pred[0]:
    plt.plot(t,y.detach().numpy())

plt.show()