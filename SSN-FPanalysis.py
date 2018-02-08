#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:32:37 2018

@author: jantinebroek
"""


import scipy as sp
#import scipy.optimize 
#from scipy import integrate
from scipy.optimize import broyden1

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt



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

# Define function only for u
def df_sol(u):
    du = ((-u + V_rest) + sp.dot(Ww, (Kk*ReLU(u - V_rest)**Nn)) + h)/tau
    return du


######################################
######    Fixed Point Analysis   #####
######################################


T_final2 = 10
Tt2 = sp.arange(T_init, T_final2, dt)

# integrate ode
u_solve = sp.integrate.odeint(df, u_0, Tt2)

x, y = u_solve.T

# plot dynamics and Phase space
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(Tt2, u_solve)
plt.xlabel('time')
plt.ylabel('V_E/V_I')
plt.title('Dynamics in time')
plt.legend(loc = 'best')

plt.subplot(1,2,2)
plt.plot(x,y, color="blue")
plt.xlabel('V_E')
plt.ylabel('V_I')
plt.title('Phase space')

plt.show()



########################################################################
######    Non-linear solver using Broyden's First Jacobian approx  #####
########################################################################

# Loop over initial values and sol for different guesses
fix_pts = sp.zeros((1000, 2))
sol = sp.zeros((2))
for idx in range(100):
    sp.random.seed(idx)
    guess = sp.random.uniform(-100, 100, 2) #automatically generates 2D array
    
    try:
        sol = broyden1(df_sol, guess, verbose = 1, maxiter = 1000)
    except:
        pass
    
    
    fix_pts[idx,:] = sol 
    print(sol)    

plt.figure(2)
plt.scatter(fix_pts[:,0], fix_pts[:,1]) #first axis is v_E, second axis is v_I
plt.show




u_E, u_I = -60., -80.

guess = sp.array([u_E, u_I])
sol = broyden1(df_sol, guess, verbose=1)

print(sol)
# add scatter plot



#increase h to see whether fixed points are the ssame
# look ate different fixed point in scatterplot for different h





###################################################
#### Calculate eigenvalues form Jacobian matrix ###
###################################################