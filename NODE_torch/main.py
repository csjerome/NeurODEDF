 #%%
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from dataset import create_dataset
from net import *
from train import *

train, test = create_dataset(dt = 0.5, time_horizon = 50)

model_phy_option = 'incomplete'
model_aug_option = True

if model_phy_option == 'incomplete':
    model_phy = Pont_roulantPDE(is_complete=False, real_params=None)
elif model_phy_option == 'complete':
    model_phy = Pont_roulantPDE(is_complete=True, real_params=None)
elif model_phy_option == 'true':
    model_phy = Pont_roulantPDE(is_complete=True, real_params=train.dataset.params)
elif model_phy_option == 'data_driven':
    model_phy = None

model_phy = Pont_roulantPDE(is_complete=False, real_params= train.dataset.params)

if model_aug_option == True :
    model_aug = MLP(state_c=4, hidden=100,input=1)
else : model_aug = None

net = Forecaster(model_phy=model_phy, model_aug=model_aug)

lambda_0 = 1.0
tau_1 = 1e-3
tau_2 = 1
niter = 5
min_op = 'l2'

#LBFGS

optimizer = torch.optim.Adam(net.parameters(), lr=tau_1, betas=(0.9, 0.999))
experiment = Experiment(
    train= train, test=test, net=net, optimizer=optimizer,
    min_op=min_op, lambda_0=lambda_0, tau_2=tau_2, niter=niter, nlog=10,
    nupdate=50, nepoch=700)

experiment.run()
#%%
