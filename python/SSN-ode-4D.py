"""
Created on Mon Feb 12 12:07:08 2018

@author: jantinebroek
"""

import scipy as sp

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

#from PyQt5 import QtCore
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure
#import matplotlib.pyplot as plt




##############################
######     Parameters    #####
##############################


# Parameters
Kk = 0.01
Nn = 2.2
V_rest = -70.

# Connectivity Matrix W
w_EE = .017
w_EP = -.956
w_EV = -.045 
w_ES = -.512
w_PE = .8535 
w_PP = -.99 
w_PV = -.09 
w_PS = -.307
w_VE = 2.104 
w_VP = -.184 
w_VV = 0 
w_VS = -.734
w_SE = 1.285 
w_SP = 0 
w_SV = -.14 
w_SS = 0

Ww = sp.array([[w_EE, w_EP, w_EV, w_ES,], [w_PE, w_PP, w_PV, w_PS], 
               [w_VE, w_VP, w_VV, w_VS], [w_SE, w_SP, w_SV, w_SS]])

# Membrane time constants
tau_E = 20/1000; #ms; 20ms for E
tau_P = 10/1000; #ms; 10ms for all PV
tau_V = 10/1000; #ms; 10ms for all VIP
tau_S = 10/1000; #ms; 10ms for all SOM
tau = sp.array([tau_E, tau_P, tau_V, tau_S])

# external forcing
h = sp.ones(4)*0

# initial and time vector
u_0 = sp.array([-60, -80, -80, -80])
T_init = 0
T_final = 100
dt = 1e-3

# noise process is Ornstein-Uhlenbeck
tau_noise = 0.05

sigma_0E = 0.2 #mV; E cells
sigma_0P = 0.1 #mV; P cells
sigma_0V = 0.1 #mV; V cells
sigma_0S = 0.1 #mV; S cells
sigma_0 = sp.array([sigma_0E, sigma_0P, sigma_0V, sigma_0S])
sigma_scaled  = sigma_0*(1. + tau/tau_noise)**0.5

eta_init = sp.array([1, -1, -1, -1])
seed = 1



##############################
######     Functions     #####
##############################



def ReLU(x, alpha=0):
    return sp.maximum(x, x*alpha*sp.ones(x.shape))

def df(u, t):
    du = ((-u + V_rest) + sp.dot(Ww, (Kk*ReLU(u - V_rest)**Nn)) + h)/tau
    return du

def euler(u, t, dt, df):
    return u + df(u, t)*dt



##############################
######  Data structures  #####
##############################



Tt = sp.arange(T_init, T_final, dt)
Uu = sp.zeros((len(Tt), 4))
Uu2 = sp.zeros((len(Tt), 4))
eta = sp.zeros((len(Tt), 4))



##############################
######    Integration    #####
##############################

# Generate a graph of fluctuations versus input
stds = []
mean = []
rate = []
h_range = sp.arange(0, 20, 0.5)
#h_range = sp.arange(1)
for h_factor in h_range:

    # update input
    print(h_factor)
    h = sp.ones(4)*h_factor


    # First generate noise vector
    eta[0, :] = eta_init
    sp.random.seed(seed)
    for iT, t in enumerate(Tt[:-1]):
        dt = Tt[iT + 1] - Tt[iT]
        eta[iT + 1, :] = eta[iT, :]  - eta[iT, :]*dt/tau_noise +\
                            sp.random.normal(0, sigma_scaled)*\
                            (2.*dt/tau_noise)**0.5

    # Next, integrate neural system with noise forcing
    Uu[0, :] = u_0
    for iT, t in enumerate(Tt[:-1]):
        dt = Tt[iT + 1] - Tt[iT]
        Uu[iT + 1, :] = euler(Uu[iT, :], t, dt, df) + eta[iT, :]*dt/tau

    # Get std and mean Vm
    stds.append(sp.std(Uu, axis=0))
    mean.append(sp.mean(Uu, axis=0))

    # Get the rates
    R = Kk*ReLU(Uu - V_rest)**Nn
    rate.append(sp.mean(R, axis=0))


plt.figure(1)

plt.subplot(3,1,1)
plt.plot(h_range, rate)
plt.ylabel('mean rate')
plt.xlabel('h')
plt.legend(['E','P','V','S'])

plt.subplot(3,1,2)
plt.plot(h_range, mean)
plt.ylabel('mean V_E/V_I [mV]')
plt.xlabel('h')
plt.legend(['E','P','V','S'])

plt.subplot(3,1,3)
plt.plot(h_range, stds)
plt.ylabel('std. dev. V_E/V_I')
plt.xlabel('h')
plt.legend(['E','P','V','S'])



#########################################################
######    Voltage output for h-factors 0, 2 and 15   #####
##########################################################
plt.figure(2)
for idx, h_factor in enumerate([0, 2, 15]):
    Uu2[0,:] = u_0

    # update input
    print(h_factor)
    h = sp.ones(2)*h_factor
    
    for iT, t in enumerate(Tt[:-1]):
        dt = Tt[iT + 1] - Tt[iT]
        Uu2[iT + 1, :] = euler(Uu2[iT, :], t, dt, df) + eta[iT, :]*dt/tau


    plt.subplot(1, 3, 1 + idx)
    plt.plot(Tt, Uu2)
    plt.ylabel('V_E/V_I [mV]')
    plt.xlabel('time')



plt.show()















