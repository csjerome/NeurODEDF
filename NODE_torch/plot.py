import matplotlib.pyplot as plt
import torch
import numpy as np
from dataset import create_dataset
from net import *
from train import *
import shelve
import matplotlib as mpl
mpl.use('TkAgg')

file = 'model_1.93e-02.pt'

param = torch.load('./exp/'+ file)
dt, horizon = param['dt'], param['horizon']
train, test = create_dataset(dt = dt, time_horizon= horizon)

'''
model_phy_option = 'complete'
model_aug_option = False

if model_phy_option == 'incomplete':
    model_phy = Pont_roulantPDE(dt = dt,is_complete=False, real_params=None)
elif model_phy_option == 'complete':
    model_phy = Pont_roulantPDE(dt = dt, is_complete=True, real_params=None)
elif model_phy_option == 'true':
    model_phy = Pont_roulantPDE(dt= dt,is_complete=True, real_params=train.dataset.params)
elif model_phy_option == 'data_driven':
    model_phy = None

if model_aug_option == True :
    model_aug = MLP(state_c=4, hidden=100,input=1)
else : model_aug = None

#model_phy = Pont_roulantPDE(dt =dt,is_complete=False, real_params= train.dataset.params)
'''

model_phy = param['model_phy']
model_aug = param['model_aug']
net = Forecaster(model_phy=model_phy, model_aug=model_aug)

net.load_state_dict(param['model_state_dict'])
net.eval()

#data = shelve.open('./_test.bak')
index = 5
data = next(iter(test))
Y = data['states'][index]
t = data['t'][index]
u = data['actions'][index]
plt.figure()
plt.plot(t,u, label= 'action')
y0 = torch.unsqueeze(Y[:, 0],0)
u = torch.unsqueeze(u,0)
pred = net(y0,t,u)

label = ['théta','dthéta',"x",'dx']
for i in range(len(Y)) :
    plt.figure()
    plt.plot(t,Y[i], label = 'data state {}'.format(label[i]))
    plt.plot(t,pred[0,i].detach().numpy(), label = 'pred state {}'.format(label[i]))
    plt.legend()
    plt.grid()


plt.show()
