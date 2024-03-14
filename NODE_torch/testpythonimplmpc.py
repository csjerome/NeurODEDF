import numpy as np
import matplotlib.pyplot as plt
import random as rd
import torch
from net import *


# conditions initiales et trajectoire de référence
condinit = [-1, 0]
Xreference = [1]*1000
# paramètres du pendule
omega0_square = 1
alpha = 0.1
sat = 1 
# paramètres du temps
Tf, dt = 5, 0.1
# paramètres de la cross entropy
Nsignaux, Horizon, Nkeptsignals, Nstepcrossentropy = 100, 10, 10, 10
# paramètres de pondération du coût
q, r = 10, 1

file = 'model_9.27e-01.pt'
param = torch.load('./exp/'+ file)
net = Forecaster(model_phy=None, model_aug=MLP(state_c=2, hidden=100,input=1))
net.load_state_dict(param['model_state_dict'])
net.eval()

def generatesignals(n, horizon, listemu, listesigma):
    L = [[0 for i in range(horizon)] for j in range(n)]
    for i in range(n):
        for j in range(horizon):
            L[i][j] = rd.normalvariate(listemu[j], listesigma[j])
    return L

def cost(xexpected, xreference, u):
    L = [xexpected[i]-xreference[i] for i in range(len(xexpected))]
    Lp = [i**2 for i in L]
    return 1/2*q*np.sqrt(sum(Lp)) #+ 1/2*r*np.linalg.norm(u)

def keepbest(signauxtest, nkeptsignals, indexlist):
    L = [signauxtest[indexlist[i]] for i in range(nkeptsignals)]
    return L

def modelependule(x, u):
    y = [x[0] + dt*x[1], x[1] + dt*(-omega0_square * np.sin(x[0]) - alpha * x[1] + u)]
    return y

def modelependulenn(x, u):
    A = np.array(param['model_state_dict']['model_aug.net.0.weight'])
    B = np.array(param['model_state_dict']['model_aug.net.0.bias'])
    C = np.array(param['model_state_dict']['model_aug.net.2.weight'])
    D = np.array(param['model_state_dict']['model_aug.net.2.bias'])
    E = np.array(param['model_state_dict']['model_aug.net.4.weight'])
    F = np.array(param['model_state_dict']['model_aug.net.4.bias'])
    p = np.array([x[0], x[1], u])
    etape1 = np.add(np.matmul(A,p),B)
    etape2 = np.add(np.matmul(C,etape1),D)
    etape3 = np.add(np.matmul(E,etape2),F)
    a = net.model_aug(torch.Tensor([x[0], x[1], u]))
    y = [x[0] + dt*a[0],x[1] + dt*a[1]]
    return y

def average(L):
    a = 0
    for i in range(len(L)):
        a = a + L[i]
    return a/len(L)

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
Statevector = condinit
posi, vite, consigne = [Statevector[0]], [Statevector[1]], []
for i in range(int(Tf/dt)):
    print(str(i)+'/'+str(int(Tf/dt)))
    mu = [0] * Horizon
    sigma = [1] * Horizon
    Testresult = [[[Statevector[0], Statevector[1]] for i in range(Horizon)] for i in range(Nsignaux)]

    for l in range(Nstepcrossentropy):
        if l != 0 :
            Testsignals = Bettersignals + generatesignals(Nsignaux - Nkeptsignals, Horizon, mu, sigma)
        else :
            Testsignals = generatesignals(Nsignaux, Horizon, mu, sigma)
        for j in range(len(Testsignals)):
            for k in range(len(Testsignals[j])):
                if Testsignals[j][k] > sat :
                    Testsignals[j][k] = sat
                elif Testsignals[j][k] < -sat :
                    Testsignals[j][k] = -sat
        for j in range(Nsignaux):
            for k in range(1, Horizon):
                Testresult[j][k] = modelependulenn(Testresult[j][k-1], Testsignals[j][k-1])
    
        Costlist = []
        for j in range(Nsignaux):
            Xexpected = [Testresult[j][k][0] for k in range(Horizon)]
            Costlist.append(cost(Xexpected, Xreference[i:i + Horizon], Testsignals[j]))
        Indexlist = sorted(enumerate(Costlist), key = lambda  x : x[1])
        Indexlist = [j[0] for j in Indexlist]
        Bettersignals = keepbest(Testsignals, Nkeptsignals, Indexlist)
        if not(verifordre([Costlist[Indexlist[i]] for i in range(len(Indexlist))])):
            print('ERREUR ERREUR ERREUR')
        mu, sigma = [], []
        if l == 0 :
            scoredeb.append(average(Costlist))
        if l == Nstepcrossentropy-1 :
            scorefin.append(average(Costlist))
        for j in range(Horizon):
            mu.append(np.mean([Bettersignals[k][j] for k in range(Nkeptsignals)]))
            sigma.append(np.std([Bettersignals[k][j] for k in range(Nkeptsignals)])) #Calcul des nouvelles moyennes et écarts-type
    Statevector = modelependule(Statevector, Bettersignals[1][1])
    posi.append(Statevector[0])
    vite.append(Statevector[1])
    consigne.append(Bettersignals[1][1])

time = [i*dt for i in range(int(Tf/dt)+1)]
timec = [i*dt for i in range(int(Tf/dt))]
ref = [Xreference[i] for i in range(len(time))]
plt.plot(time, posi, color='blue', label = 'Position')
plt.plot(time, vite, color='red', label = 'Vitesse')
plt.plot(timec, consigne, color ='green', label = 'Consigne')
plt.plot(time, ref, color='black', label = 'Référence')
plt.legend()
plt.show()

plt.plot(scoredeb, color = 'blue')
plt.plot(scorefin, color = 'red')
plt.show()