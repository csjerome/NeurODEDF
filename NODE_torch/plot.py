import matplotlib.pyplot as plt

from dataset import create_dataset
from net import *
from train import *
import matplotlib as mpl
import tikzplotlib # used for latex plot

file = 'model_1.52e-01.pt'

## Plot style and option ##
mpl.use('TkAgg')
plt.style.use('seaborn-v0_8')
params = {"font.serif" : ["Computer Modern Serif"],}
plt.rcParams.update(params)

## Load designated file and extract model ##
param = torch.load('./exp/'+ file)
dt, horizon = param['dt'], param['horizon']
model_phy = param['model_phy']
model_aug = param['model_aug']
net = Forecaster(model_phy=model_phy, model_aug=model_aug)
net.load_state_dict(param['model_state_dict'])
net.eval()

## Create dataset (or import it from existing file) ##
train, test = create_dataset(dt = dt, time_horizon= horizon)

index = 9 # index of the data used for plotting
data = next(iter(test))
Y = data['states'][index]
t = data['t'][index]
u = data['actions'][index]

## Plot input ##
plt.figure()
plt.plot(t,u, label= 'action')
plt.show()
y0 = torch.unsqueeze(Y[:, 0],0)
u = torch.unsqueeze(u,0)
pred = net(y0,t,u)

## Run model and plot ##
label = ['théta','dthéta',"x",'dx']
fig, axs = plt.subplots(2,2)
for i in range(len(Y)) :
    axs[i//2,i%2].plot(t,Y[i], label = 'data state {}'.format(label[i]))
    axs[i//2,i%2].plot(t,pred[0,i].detach().numpy(), label = 'pred state {}'.format(label[i]))
    axs[i//2,i%2].grid()
    axs[i//2,i%2].set_title('Evolution de l\'état {} prédite comparée aux données'.format(label[i]))
    axs[i//2,i%2].legend()

#tikzplotlib.save('node')
plt.show()

#%%
