import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import tensorflow.keras as kr
import tensorflow.keras.optimizers as tfko
from tfdiffeq import odeint
import keyboard as kb
import os
import configparser

#%%
## ------------   Files and Parameters   --------------- ##

paramfile = "default.txt"

cg = configparser.ConfigParser()
cg.read(paramfile)
w0 = float(cg.get('System Parameters','w0'))
f = float(cg.get('System Parameters','f'))
x0 = float(cg.get('System Parameters','x0'))
y0 = float(cg.get('System Parameters','y0'))

w0i = float(cg.get('Initial Values', 'w0i'))
fi = float(cg.get('Initial Values', 'fi'))

t_0 = float(cg.get('Train Values', 't_0'))
t_end = float(cg.get('Train Values', 't_end'))
n_samples = int(cg.get('Train Values', 't_sample'))
learning_rate = float(cg.get('Train Values','learning_rate'))
epochs = int(cg.get('Train Values','epochs'))

true_params = [w0, f]
s = np.sqrt(1-f**2)

an_sol_x = lambda t : y0/w0/s*np.exp(-w0*f*t)*np.sin(w0*s*t)
an_sol_y = lambda t : y0/s*np.exp(-w0*f*t)*(s*np.cos(w0*s*t)+f*np.sin(w0*s*t))

t_space =np.linspace(t_0, t_end, n_samples,dtype = "float32")

dataset_outs = tf.expand_dims(np.transpose([an_sol_x(t_space), an_sol_y(t_space)]), axis = 2)
t_space_tensor = tf.constant(t_space)
x_init = tf.constant([x0], dtype=t_space_tensor.dtype)
y_init = tf.constant([y0], dtype=t_space_tensor.dtype)
u_init = tf.convert_to_tensor([x_init, y_init], dtype=t_space_tensor.dtype)
args = [tf.Variable(initial_value=w0i, name='w0', trainable=True,dtype=t_space_tensor.dtype),
        tf.Variable(initial_value=fi, name='f', trainable=True,dtype=t_space_tensor.dtype)]

optimizer = tfko.Adam(learning_rate=learning_rate)

#%%
## ------------- Functions ----------------- ##

def parametric_ode_system(t, u, args):
    w , k= args[0], args[1]
    x, y = u[0], u[1]
    dx_dt = y
    dy_dt = -x*w**2 - 2*k*w*y
    return tf.stack([dx_dt, dy_dt])

class MLP(kr.Model):
    def __init__(self, hidden):
        super().__init__()
        self.net = tf.keras.models.Sequential()
        self.net.add(kr.Input(shape = (2,)))
        self.net.add(kr.layers.Dense(hidden, activation = 'relu'))
        self.net.add(kr.layers.Dense(hidden, activation= 'relu'))
        self.net.add(kr.layers.Dense(2))



    def __call__(self, t, state):
        return tf.convert_to_tensor(np.transpose(self.net(np.transpose(state))), dtype=t_space_tensor.dtype)


class Net(kr.Model):
    def __init__(self,hidden):
        super().__init__()
        self.model = MLP(hidden)
        #print(self.model(t_space_tensor, u_init))

    def __call__(self, u_init,t):
        return odeint(self.model,u_init,t_space_tensor)
'''
def net():
    return odeint(lambda ts, u0: parametric_ode_system(ts, u0, args),
                  u_init, t_space_tensor)
'''

def loss_func(num_sol):
    return tf.reduce_sum(tf.square(dataset_outs[0] - num_sol[:, 0])) + \
        tf.reduce_sum(tf.square(dataset_outs[1] - num_sol[:, 1]))


#%%

## ------------------- Training ------------------ ##
L = []
#Prm = []


epoch = 0
net = Net(hidden = 100)
loss = kr.losses.MeanSquaredError()

net.compile(
    optimizer=optimizer,
    loss= loss)

while epoch < epochs and not kb.is_pressed("alt+ctrl+shift"):
    with tf.GradientTape() as tape:
        tape.watch(net.model.trainable_weights)
        num_sol = net(u_init,t_space_tensor)
        loss_value = loss(dataset_outs,num_sol)
        print([np.shape(_) for _ in net.model.trainable_weights])
        print("Epoch:", epoch, " loss:", loss_value.numpy())
        L.append(loss_value.numpy())
        grads = tape.gradient(loss_value, net.model.trainable_weights)
        print(grads)
        optimizer.apply_gradients(zip(grads, net.model.trainable_weights))
        epoch +=1

#print("Learned parameters:", [arg.numpy() for arg in args])

#%%


## ----------------------- Plots ---------------------- ##
mpl.use('TkAgg')
num_sol = net()
x_num_sol = num_sol[:, 0].numpy()
y_num_sol = num_sol[:, 1].numpy()

x_an_sol = an_sol_x(t_space)
y_an_sol = an_sol_y(t_space)

T = np.arange(0,epoch)
plt.figure()
plt.plot(T, L)
plt.title('Loss evolution')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.grid()


plt.figure()
plt.scatter(np.array(Prm)[:,0], np.array(Prm)[:,1], c = T*2/epoch -1)
plt.scatter(true_params[0],true_params[1], c = 'r')
plt.title("Parameters evolution")
plt.xlabel(args[0].name[:-2])
plt.ylabel(args[1].name[:-2])
plt.grid()

plt.figure()
plt.plot(t_space, x_an_sol,'--', linewidth=2, label='analytical x')
plt.plot(t_space, y_an_sol,'--', linewidth=2, label='analytical y')
plt.plot(t_space, x_num_sol, linewidth=1, label='numerical x')
plt.plot(t_space, y_num_sol, linewidth=1, label='numerical y')
plt.title('Neural ODEs to fit params')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.show()

#%%

