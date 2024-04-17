import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

from dataset import create_dataset
from net import *
from train import *
from PIL import ImageGrab
import matplotlib as mpl
import tikzplotlib # used for latex plot

file = 'model_4.99e-02.pt'
save_to_gif = True

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

y0 = torch.unsqueeze(Y[:, 0],0)
u = torch.unsqueeze(u,0)
pred = net(y0,t,u)

import tkinter as tk
from math import pi, sin, cos

G = 9.81

class SimplePendulum():
    def __init__(self, real, length, initial_theta):
        self.length = length
        self.initial_theta = initial_theta
        self.time = 0
        self.real = real
        self.frame = tk.Frame(root)
        self.scl_length = tk.Scale(self.frame, from_=25, to=350, length= 300, tickinterval=50, label= 'Length of Pendulum', troughcolor='yellow', orient=tk.HORIZONTAL)
        self.scl_length.set(length)

    def grid_widgets(self):
        self.frame.grid(row=0, column=1, rowspan=3)
        self.scl_length.grid(row=0,column=0, padx=10, pady=5)

    def incr_theta(self):
        self.initial_theta += pi/8
        if self.initial_theta >= pi:
            self.initial_theta = pi - pi/8
        self.time = 0

    def decr_theta(self):
        self.initial_theta -= pi/8
        if self.initial_theta < 0:
            self.initial_theta = 0
        self.time = 0

    def update_pendulum(self, pivot_x, pivot_y):
        self.length = self.scl_length.get()
        if self.real :
            self.theta = Y[0,int(self.time/dt)].numpy()
        else :
            self.theta = pred[0,0,int(self.time/dt)].detach().numpy()

        pend_x = pivot_x + self.length * sin(self.theta)
        pend_y = pivot_y + self.length * cos(self.theta)
        self.time += dt
        if self.time > horizon:
            self.time = 0
        return pend_x, pend_y


class MainApplication(tk.Tk):
    def __init__(self, master, pendulum_params):
        self.master = master
        self.height = 800
        self.width = 800
        self.frm_canvas = tk.Frame(self.master)
        self.canvas = tk.Canvas(self.frm_canvas, height=self.height, width=self.width, bg="black")

        self.pendulum = SimplePendulum(True, **pendulum_params)
        self.data_pendulum = SimplePendulum(False,**pendulum_params)
        self.pivot_x = int(self.canvas['width'])/2 # Location of pendulum pivot on canvas
        self.pivot_y = 300
        self.bool_trace = False
        self.bool_pause = False
        self.all_traces = []
        self.curr_trace = []
        self.grid_pack_widgets()
        self.draw_simple_pendulum()


    def grid_pack_widgets(self):
        self.frm_canvas.grid(row=0, column=0)
        self.canvas.pack()
    def draw_simple_pendulum(self):
        self.canvas.delete("pendulum")
        self.canvas.delete("line")
        self.canvas.delete("trace")
        self.simple_motion()
        tk.after_id = self.canvas.after(15, self.draw_simple_pendulum)


    def simple_motion(self):
        (x, y) = self.pendulum.update_pendulum(self.pivot_x, self.pivot_y)
        radius = 20
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, tag="pendulum", fill="white")
        self.canvas.create_line(self.pivot_x, self.pivot_y, x, y, width=3, tag="line", fill="white")

        if self.bool_trace:
            self.curr_trace.append((x,y,x,y))
        if self.all_traces:
            for trace in self.all_traces:
                self.canvas.create_line(trace, tag="trace", fill="white")
        if self.curr_trace:
            self.canvas.create_line(self.curr_trace, tag="trace", fill="white")

        (x, y) = self.data_pendulum.update_pendulum(self.pivot_x, self.pivot_y)
        radius = 20
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, tag="pendulum", fill="cyan")
        self.canvas.create_line(self.pivot_x, self.pivot_y, x, y, width=3, tag="line", fill="cyan")

        if self.bool_trace:
            self.curr_trace.append((x,y,x,y))
        if self.all_traces:
            for trace in self.all_traces:
                self.canvas.create_line(trace, tag="trace", fill="white")
        if self.curr_trace:
            self.canvas.create_line(self.curr_trace, tag="trace", fill="white")

        if save_to_gif:
            savename = 'images_gif/im_{0:.2f}'.format(self.pendulum.time)
            ImageGrab.grab((20,50,self.width+150,self.height+150)).save(savename + '.jpg') # dimension are hand-tuned

    def start_trace(self):
        if self.curr_trace:
            self.all_traces.append(self.curr_trace)
        self.curr_trace = []
        self.bool_trace = not self.bool_trace

    def clear_trace(self):
        self.curr_trace = []
        self.all_traces = []
        self.canvas.delete('trace')


if __name__ == '__main__':

    pend_params = {'length' : 1000,
                   'initial_theta' : y0[0]}

    root = tk.Tk()
    root.geometry('800x800+0+0')
    root.title("Pendulum Simulation")
    root.resizable(False,False)
    root.columnconfigure(1, weight=1)
    app = MainApplication(root, pend_params)
    root.mainloop()