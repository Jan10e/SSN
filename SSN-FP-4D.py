#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:07:08 2018

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

Ww = sp.array([[w_EE, w_EP, w_EV, w_ES],
    [w_PE, w_PP, w_PV, w_PS],
    [w_VE, w_VP, w_VV, w_VS],
    [w_SE, w_SP, w_SV, w_SS]])


# Membrane time constants
tau_E = 0.02
tau_P = 0.01
tau_S = 0.01
tau_V = 0.01
tau = sp.array([tau_E, tau_P, tau_S, tau_V])

# external forcing
h = sp.ones(4)*0

# initial and time vector
u_0 = sp.array([-80, -60, -60, -60])
T_init = 0
T_final = 100
dt = 1e-3

# noise process is Ornstein-Uhlenbeck
tau_noise = 0.05
sigma = sp.array([0.2, 0.1, 0.1, 0.1])
sigma_scaled = sigma*(1. + tau/tau_noise)**0.5
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

# Define function only for u
def df_sol(u):
    du = ((-u + V_rest) + sp.dot(Ww, (Kk*ReLU(u - V_rest)**Nn)) + h)/tau
    return du



########################################################################
######    Non-linear solver using Broyden's First Jacobian approx  #####
########################################################################

# For single initial points
u_E, u_P, u_S, u_V = -80., -60., -60., -60.

guess = sp.array([u_E, u_P, u_S, u_V])
sol = broyden1(df_sol, guess, verbose=1)

print(sol)




# Loop over initial values and sol for different guesses
seeds = range(0,10)
fix_pts = sp.zeros((10, 4))
for idx in seeds:
    sp.random.seed(idx)
    while True:
        try:
            guess = sp.random.uniform(-100, 100, 4)
            sol = broyden1(df_sol, guess, verbose = 0, maxiter = 100)
        except:
            continue 
        else: 
            fix_pts[idx,:] = sol 
            print(sol) 
            break
            

plt.figure(2)
plt.scatter(fix_pts[:,0], fix_pts[:,1]) #first axis is v_E, second axis is v_P

plt.figure(3)
plt.scatter(fix_pts[:,0], fix_pts[:,2]) #first axis is v_E, second axis is v_S

plt.figure(4)
plt.scatter(fix_pts[:,0], fix_pts[:,3]) #first axis is v_E, second axis is v_V

plt.figure(5)
plt.scatter(fix_pts[:,1], fix_pts[:,2]) #first axis is v_P, second axis is v_S

plt.figure(6)
plt.scatter(fix_pts[:,1], fix_pts[:,3]) #first axis is v_P, second axis is v_V

plt.figure(7)
plt.scatter(fix_pts[:,2], fix_pts[:,3]) #first axis is v_S, second axis is v_V


plt.show


#look at different fixed point in scatterplot for different h
# h = 0 gives fixed point (-70, -70, -70, -70)
# h = 2 gives fixed points (-68, -68, -67, -67)
# h = 15 gives fixed point (-60, -58, -55, -53)



###################################################
#### Approximate eigenvalues from Jacobian matrix ###
###################################################
# to look whether the fixed points are stable or unstable
from scipy.optimize import fsolve


guess = sp.random.uniform(-100, 100, 4)

#Get Jacobian approximate
x, infodict, ier, mesg = fsolve(df_sol, guess, full_output = True)

print(x)
print(infodict.keys())

# Get the Jacobian by matrix multiplication
Q = infodict['fjac']
r = infodict['r']
R = sp.array([[r[0], r[1], r[2], r[3]], 
              [0, r[4], r[5], r[6]],
              [0, 0, r[7], r[8]],
              [0, 0, 0, r[9]]])
J = sp.dot(Q,R)

# Get eigenvalues from Jacobian
eigJ = sp.linalg.eig(J)
eig_val = eigJ[0]
eig_vec = eigJ[1]

print(eig_val)
#print(eig_vec)

#h = 0: eigenvalues are all negative, therefore fixed point is -70 and is an attractive  
#stable fixed point

#h = 2: eigenvalues are all negative, with pos imaginary values for P and neg imaginary value for S

#h = 15: eigenvalues are all negative, with pos img value for E and neg img value for P







