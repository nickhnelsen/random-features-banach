# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:56:24 2020

@author: nickh
"""

import numpy as np

# Activation Functions
def relu(r):
    """ Rectified Linear Unit (ReLU) activation function """
    return np.maximum(0,r)

def elu(r, alpha=1):
    """ Exponential Linear Unit (eLU) activation function """
    return np.where(r < 0, alpha * (np.exp(r) - 1), r)

def lrelu(r, alpha=1e-2):
    """ Leaky relu """
    return np.where(r < 0, alpha*r, r)

def selu(r, alpha=1.67326, lam=1.0507):
    """ 
    Scaled Exponential Linear Unit (SeLU) activation function
    -- useful for Batch Renormalization Situations according to Kovachki, Stuart IP paper 
    """
    return lam*np.where(r < 0, alpha * (np.exp(r) - 1), r)

def heaviside(r):
    """ True Heaviside step function"""
    return np.heaviside(r,0.5)

def smoothheaviside(r):
    """ Smoothed Heaviside step function"""
    ep=0.01
    return 1/2 + (1/np.pi)*np.arctan(r/ep)

def sawtooth(r):
    """ Sawtooth function from Yarotsky"""
#    return np.where(r < 1/2, np.maximum(0,2*r), np.maximum(0,2*(1-r)))
    return np.maximum(0,np.minimum(2*r,2-2*r))

def softplus(r):
    """ Softplus activation function (smoothed rectifier), defined as np.log(1+np.exp(r)) """
    return np.log(1 + np.exp(-np.abs(r))) + np.maximum(r,0)

def tansig(r):
    """ Tan-Sigmoid activation function defined as (2/(1+np.exp(-2*r))-1), equivalent to hyperbolic tangent tanh(\cdot) """
    return np.tanh(r)

def sigmoid(r):
    """ Sigmoid activation function """
    return 1/(1+np.exp(-r))