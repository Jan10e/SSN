"""
Simulate a E-I 2D model with supra-linear stabilization

Created by Nirag Kadakia at 16:51 01-29-2018
This work is licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License.
To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-sa/4.0/.
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
Kk = 0.3
Nn = 2
V_rest = -70.

# Connectivity Matrix W
w_EE = 1.25
w_EI = -0.65
w_IE = 1.2
w_II = -0.5
Ww = sp.array([[w_EE, w_EI], [w_IE, w_II]])

# Membrane time constants
tau_E = 0.02
tau_I = 0.01
tau = sp.array([tau_E, tau_I])

# external forcing
h = sp.ones(2)*0

# initial and time vector
u_0 = sp.array([-60, -80])
T_init = 0
T_final = 100
dt = 1e-3

# noise process is Ornstein-Uhlenbeck
tau_noise = 0.05
sigma = sp.array([0.2, 0.1])
sigma_scaled = sigma*(1. + tau/tau_noise)**0.5
eta_init = sp.array([1, -1])
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
Uu = sp.zeros((len(Tt), 2))
Uu2 = sp.zeros((len(Tt), 2))
eta = sp.zeros((len(Tt), 2))



##############################
######    Integration    #####
##############################

# Generate a graph of fluctuations versus input
stds = []
mean = []
rate = []
h_range = sp.arange(0, 20, 2.5)
#h_range = sp.arange(1)
for h_factor in h_range:

    # update input
    print(h_factor)
    h = sp.ones(2)*h_factor


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

plt.subplot(3,1,2)
plt.plot(h_range, mean)
plt.ylabel('mean V_E/V_I [mV]')
plt.xlabel('h')

plt.subplot(3,1,3)
plt.plot(h_range, stds)
plt.ylabel('std. dev. V_E/V_I')
plt.xlabel('h')




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





######################################
######    Fixed Point Analysis   #####
######################################
import numpy as np
from scipy import integrate

T_final2 = 10
Tt2 = sp.arange(T_init, T_final2, dt)

# integrate ode
u_solve = integrate.odeint(df, u_0, Tt2)

x, y = u_solve.T

# plot dynamics and Phase space
plt.figure(3)
plt.subplot(1,2,1)
plt.plot(Tt2, u_solve)
plt.xlabel('time')
plt.ylabel('V_E/V_I')
plt.title('Dynamics in time')
plt.legend(loc = 'best')

plt.subplot(1,2,2)
plt.plot(x,y, color="blue")
plt.xlabel('V_E')               # Correct axis labels?
plt.ylabel('V_I')
plt.title('Phase space')

plt.show()



######################################
######    Find Fixed Points   #####
######################################
fp = []

def find_fixed_points(r):
    for x in range(r):
        for y in range(r):
            if df == 0:
                fp.append((x,y))
                print('The system has a fixed point in %s,%s' % (x,y))
    return fp

find_fixed_points(10000)





import sympy as sm
# define the system in this way (assuming a predator-prey-system with no negative values)
# to avoid interference x = r (for resource) and y = c (for consumer)
v = sm.symbols('v', negative = False)
#R = 2*r - r**2 - r*c
#C = -c + r*c

# use sympy's way of setting equations to zero
u_equal = sm.Eq(df, 0)


# compute fixed points
equilibria = sm.solve(u_equal, v)
print(equilibria)






###################################################
#### Calculate eigenvalues form Jacobian matrix ###
###################################################










