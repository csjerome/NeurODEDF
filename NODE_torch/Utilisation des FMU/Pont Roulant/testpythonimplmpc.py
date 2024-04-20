import numpy as np
import matplotlib.pyplot as plt
import random as rd
import torch
from net import *
import matplotlib as mpl
mpl.use('TkAgg')

# conditions initiales et trajectoire de référence
condinit = np.array([0.3, 0.,0.,0.])
Xreference = np.ones(1000) #consigne
# paramètres du pendule
omega0_square = 1
alpha = 0.1
sat = 2
# paramètres du temps
Tf, dt = 100, 0.1
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
    y = [x[0] + dt*dy[0],x[1] + dt*dy[1]]
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


scoredeb, scorefin =[], []
Statevector = condinit.astype(np.float32)
posi, vite, consigne = [Statevector[0]], [Statevector[1]], []
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
            Xexpected = Testresult[j,1,:]
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
            sigma.append(np.std([Bettersignals[k][j] for k in range(Nkeptsignals)])) #Calcul des nouvelles moyennes et écarts-type
    net.derivative_estimator.action = torch.tensor([Bettersignals[0]])
    Statevector = net.int_(net.derivative_estimator, torch.tensor([Statevector]), t = torch.tensor([0.,dt])).detach().numpy()[1,0]
    #print(Statevector)
    posi.append(Statevector[2])
    vite.append(Statevector[0])
    consigne.append(Bettersignals[0][0])

time = [i*dt for i in range(int(Tf/dt)+1)]
timec = [i*dt for i in range(int(Tf/dt))]
ref = [Xreference[i] for i in range(len(time))]
plt.plot(time, posi, color='blue', label = 'Position')
plt.plot(time, vite, color='red', label = 'Angle')
plt.plot(timec, consigne, color ='green', label = 'Consigne')
plt.plot(time, ref, color='black', label = 'Référence')
plt.legend()
plt.show()

#plt.plot(scoredeb, color = 'blue')
#plt.plot(scorefin, color = 'red')
