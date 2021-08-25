#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import utils
import synthetic

import torch

# In[2]:

# note xx.shape=yy.shape=(n,n)


def state(xx, yy, h_sat, h):
    """Returns magnetic hysteresis state as an mxnxn tensor, where 
    m is the number of distinct applied magnetic fields. The 
    states are initially entirely off, and then updated per
    time step depending on both the most recent applied magnetic
    field and prior inputs (i.e. the "history" of the states tensor). 
    
    For each time step, the state matrix is either "swept up" or 
    "swept left" based on how the state matrix corresponds to like 
    elements in the meshgrid; the meshgrid contains alpha, beta 
    coordinates which serve as thresholds for the hysterion state to
    "flip".
    
    See: https://www.wolframcloud.com/objects/demonstrations/TheDiscretePreisachModelOfHysteresis-source.nb

    Parameters
    ----------
    h_sat : float,
        The maximum alpha, beta in the distribution.

    n : int,
        The number of points expected in the discretization.

    h : array,
        The applied magnetic field H_1:t={H_1, ... ,H_t}, where 
        t represents each time step. 

    Raises
    ------
    ValueError
        If n is negative.
    """

    hyst_state = torch.ones(xx.shape)*-1  # n x n matrix of hysterion magnetization state given (a,b)
    # starts off off
    h = torch.cat((torch.tensor([-h_sat]), torch.tensor(h)))  # H_0=-t, negative saturation limit
    states = torch.empty((len(h), xx.shape[0], xx.shape[1]))  # list of hysteresis states
    for i in range(len(h)):
        if h[i] > h[i - 1]:
            hyst_state = torch.tensor([[hyst_state[j][k] if yy[j][k] >= h[i] else 1 for k in range(len(yy[j]))] for j in range(len(yy))])
        elif h[i] < h[i - 1]:
            hyst_state = torch.tensor([[hyst_state[j][k] if xx[j][k] <= h[i] else -1 for k in range(len(yy[j]))] for j in range(len(xx))])
        hyst_state = torch.tril(hyst_state)
        states[i] = hyst_state
    return states


def discreteIntegral(xx, yy, h_sat, b_sat, dens_i, h, n):
    """Returns the resulting magnetic field as an array equal
    in size to the applied magnetic field vector. The function
    computes the Hadamard product of the state matrix with the 
    density matrix. This gives the magnetic field for each coordinate
    (alpha, beta). The sum of these values at all the points in the 
    space is the total resulting magnetic field. 
    
    Plots the resulting magnetic field by input applied magnetic
    field. If this input field h increases then decreases, the 
    plot forms a hysteresis loop.
    
    Parameters
    ----------
    n : int,
        The number of points expected in the discretization.

    h_sat : float,
        The maximum alpha, beta in the distribution.

    b_sat : float
        Magnetic B field at saturation
    
    dens_i : array, 
        A vector of dummy values for the initial density. The
        size must be n**2/2 + n/2 to populate the upper triangle,
        including the diagonal. 

    h : array,
        The applied magnetic field H_1:t={H_1, ... ,H_t}, where 
        t represents each time step. 

    Raises
    ------
    RuntimeError
        If len(dens_i) does not equal n**2/2 + n/2.
        
    ValueError
        If n is negative.
    """

    b = torch.empty(len(h))  # b is the resulting magnetic field
    states = state(xx,yy, h_sat, h)
    dens = utils.vector_to_tril(dens_i, n)
    a = b_sat / torch.sum(dens)
    for i in range(len(h)):
        # print(dens * states[i+1])
        b[i] = torch.sum(dens * states[i + 1])
    return b * a


# h = np.append(np.linspace(-1.0, 1.0, 10), np.flipud(np.linspace(-1.0, 1.0, 10)))
# n = 100
# h_sat = 1.0
# xx, yy = utils.generate_mesh(h_sat, n)
# synthetic_mu = synthetic.gaussian_density(xx, yy)
# fig, ax = plt.subplots()
# ax.pcolor(xx, yy, synthetic_mu)

# mu_vector = utils.tril_to_vector(synthetic_mu, n)
# b = discreteIntegral(xx, yy, 0.8, 2, mu_vector, h)

# plt.figure()
# plt.plot(h, b, 'o')
# plt.show()
