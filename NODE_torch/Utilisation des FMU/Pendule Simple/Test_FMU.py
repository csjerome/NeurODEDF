import numpy as np
import matplotlib.pyplot as plt
import random as rd
import torch
from net import Forecaster, MLP
from Test_PenduleSimple_DoStep import FMU
import matplotlib as mpl
mpl.use('TkAgg')

# Conditions initiales et trajectoire de référence
condinit = [-1.0, 0.0]
Xreference = [1.0] * 1000
# Paramètres du temps
Tf, dt = 5, 0.1
# Paramètres de la cross-entropy
Nsignaux, Horizon, Nkeptsignals, Nstepcrossentropy = 100, 10, 10, 10
# Paramètres de pondération du coût
q, r = 10, 1
# Valeurs de saturation
sat = 2

# Charger le modèle et le préparateur
file = 'model_9.27e-01.pt'
param = torch.load('./exp/' + file)
net = Forecaster(model_phy=None, model_aug=MLP(state_c=2, hidden=100, input=1))
net.load_state_dict(param['model_state_dict'])
net.eval()

# Modèle donné par la FMU
fmu = FMU('Test_Pendule.fmu', inputs=['In1'], outputs=['Out1'])
fmu.initialize(start_time=0.0)

def generatesignals(n, horizon, listemu, listesigma):
    L = [[0 for i in range(horizon)] for j in range(n)]
    for i in range(n):
        for j in range(horizon):
            L[i][j] = rd.normalvariate(listemu[j], listesigma[j])
    return L

def cost(xexpected, xreference, u):
    L = [xexpected[i] - xreference[i] for i in range(len(xexpected))]
    Lp = [i**2 for i in L]
    return 1 / 2 * q * np.sqrt(sum(Lp))

def keepbest(signauxtest, nkeptsignals, indexlist):
    L = [signauxtest[indexlist[i]] for i in range(nkeptsignals)]
    return L

def modelependulenn(x, t, u):
    x = torch.tensor(x) if np.isscalar(x) else torch.tensor(x).unsqueeze(0)
    u = torch.tensor(u) if np.isscalar(u) else torch.tensor(u).unsqueeze(0)
    Y = torch.cat((x, u), -1)
    dy = net.forward(x, t, u)
    y = [x[0] + dt * dy[0], x[1] + dt * dy[1]]
    return y

def average(L):
    return sum(L) / len(L)

def standarddeviation(L):
    avg = average(L)
    return np.sqrt(sum((x - avg)**2 for x in L) / len(L))

def verifordre(L):
    return all(L[i] < L[i+1] for i in range(len(L)-1))

scoredeb, scorefin = [], []
Statevector = condinit
posi, consigne = [Statevector[0]], []

for i in range(int(Tf/dt)):
    print(f"{i}/{int(Tf/dt)}")
    mu = [0] * Horizon
    sigma = [1] * Horizon
    Testresult = [[[Statevector[0]] for _ in range(Horizon)] for _ in range(Nsignaux)]

    for l in range(Nstepcrossentropy):
        if l != 0:
            Testsignals = Bettersignals + generatesignals(Nsignaux - Nkeptsignals, Horizon, mu, sigma)
        else:
            Testsignals = generatesignals(Nsignaux, Horizon, mu, sigma)

        for j in range(len(Testsignals)):
            for k in range(len(Testsignals[j])):
                if Testsignals[j][k] > sat:
                    Testsignals[j][k] = sat
                elif Testsignals[j][k] < -sat:
                    Testsignals[j][k] = -sat

        for j in range(Nsignaux):
            for k in range(1, Horizon):
                # Résultat donnés par le FMU
                Testresult[j][k] = modelependulenn(Testresult[j][k-1], i * dt, Testsignals[j][k])

        Costlist = []
        for j in range(Nsignaux):
            Xexpected = [Testresult[j][k][0] for k in range(Horizon)] 
            Costlist.append(cost(Xexpected, Xreference[i:i + Horizon], Testsignals[j]))
        
        Indexlist = sorted(enumerate(Costlist), key=lambda x: x[1])
        Indexlist = [j[0] for j in Indexlist]
        Bettersignals = keepbest(Testsignals, Nkeptsignals, Indexlist)

        if not verifordre([Costlist[Indexlist[i]] for i in range(len(Indexlist))]):
            print('ERREUR ERREUR ERREUR')

        mu, sigma = [], []
        if l == 0:
            scoredeb.append(average(Costlist))
        if l == Nstepcrossentropy-1:
            scorefin.append(average(Costlist))

        for j in range(Horizon):
            mu.append(np.mean([Bettersignals[k][j] for k in range(Nkeptsignals)]))
            sigma.append(np.std([Bettersignals[k][j] for k in range(Nkeptsignals)]))

    # Actualisation de Statevector avec le modèle donné par la FMU
    Statevector = [fmu.doStep(time=0, step_size=dt)]  
    
    posi.append(Statevector[0])  
    consigne.append(Bettersignals[1][1])

time = [i * dt for i in range(int(Tf/dt) + 1)]
timec = [i * dt for i in range(int(Tf/dt))]
ref = [Xreference[i] for i in range(len(time))]

plt.plot(time, posi, color='blue', label='Position')
plt.plot(timec, consigne, color='green', label='Commande')
plt.plot(time, ref, color='black', label='Consigne')
plt.legend()
plt.show()

plt.plot(scoredeb, color='blue')
plt.plot(scorefin, color='red')
plt.show()