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
h = sp.ones(2)*2

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

# For single initial points
u_E, u_I = -60., -80.

guess = sp.array([u_E, u_I])
sol = broyden1(df_sol, guess, verbose=1)

print(sol)




# Loop over initial values and sol for different guesses
seeds = range(0,10)
fix_pts = sp.zeros((10, 2))
for idx in seeds:
    sp.random.seed(idx)
    while True:
        try:
            guess = sp.random.uniform(-100, 100, 2)
            sol = broyden1(df_sol, guess, verbose = 0, maxiter = 100)
        except:
            continue 
        else: 
            fix_pts[idx,:] = sol 
            print(sol) 
            break
            

plt.figure(2)
plt.scatter(fix_pts[:,0], fix_pts[:,1]) #first axis is v_E, second axis is v_I
#plt.ylim([-80, -60])
#plt.xlim([-80, -60])
plt.show


#look at different fixed point in scatterplot for different h
# h = 0 gives fixed point (-70, -70)
# h = 2 gives fixed points (-66, -66)
# h = 15 gives fixed point (-64, -59)



###################################################
#### Approximate eigenvalues from Jacobian matrix ###
###################################################
# to look whether the fixed points are stable or unstable
from scipy.optimize import fsolve


guess = sp.random.uniform(-100, 100, 2)

#Get Jacobian approximate
x, infodict, ier, mesg = fsolve(df_sol, guess, full_output = True)

print(x)
print(infodict.keys())

# Get the Jacobian by matrix multiplication
Q = infodict['fjac']
r = infodict['r']
R = sp.array([[r[0], r[1]], [0, r[2]]]).T
J = sp.dot(Q,R)

# Get eigenvalues from Jacobian
eigJ = sp.linalg.eig(J)
eig_val = eigJ[0]
eig_vec = eigJ[1]

#eigenvalues are both negative, therefore fixed point (-70 for h=0) is attractive and 
#a stable fixed point



###########################################################
######     Algorithmically solving for eigenvalues    #####
###########################################################

def dFE_dVE(V0):
    VE = V0[0]
    VI = V0[1]
    dJ1 = (-1 + w_EE * 2 * (VE - V_rest))/tau_E
    return dJ1

def dFE_dVI(V0):
    VE = V0[0]
    VI = V0[1]
    dJ2 = (w_EI * 2 * (VE - V_rest))/tau_E
    return dJ2    

def dFI_dVE(V0):
    VE = V0[0]
    VI = V0[1]
    dJ3 = (w_IE * 2 * (VI - V_rest))/tau_I
    return dJ3 

def dFI_dVI(V0):
    VE = V0[0]
    VI = V0[1]
    dJ4 = (-1 + w_II * 2 * (VI - V_rest))/tau_I
    return dJ4 

# input V_0, which is fixed point
V0 = fix_pts[0,:] 

# Jacobian
Jb = sp.array([[dFE_dVE(V0), dFE_dVI(V0)],[dFI_dVE(V0), dFI_dVI(V0)]])

# Get eigenvalues from Jacobian
eigJb = sp.linalg.eig(Jb)
eig_val_Jb = eigJb[0]
eig_vec_Jb = eigJb[1]

#comparable results with approximate for eignevalues: matrix is [[-1, 0], [0, -1]]

#eigenvalues are both negative, therefore fixed point (-70 for h=0) is attractive and 
#a stable fixed point



