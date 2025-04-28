#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:27:56 2025

@author: rrowekamp
"""

from keras.models import Model
from numpy import dot,float32,float64,ndarray,prod,rot90
from numpy.linalg import eigh

def sparseness(x    : ndarray
               ) -> float32 | float64:
    """
    Calculates the sparsenesss of the given array.

    Parameters
    ----------
    x : ndarray
        An array of values to analyze.

    Returns
    -------
    float32 | float64
        The sparseness of the given array. The minimum sparseness is zero for 
        the case of a uniform array. The maximum value is N - 1, where N is the 
        size of the array.

    """
    
    return x.var()/x.mean()**2

def rotate_layer1(model     : Model,
                  ) -> Model :  
    """
    Calculates the response of the model if the first layer is rotated 90 
    degrees.

    Parameters
    ----------
    model : Model
        Model to be modified.

    Returns
    -------
    Model
        The modified model.

    """
    
    weights = model.get_weights()
    
    weights[0] = rot90(weights[0],axes=(1,2))
    weights[2] = rot90(weights[2],axes=(1,2))
    
    model.set_weights(weights)
    
    return model

def rotate_suppression(model    : Model,
                       layer    : int
                       ) -> Model :
    """
    Rotates the suppressive component of the quadratic term of the selected 
    layer 90 degrees.

    Parameters
    ----------
    model : Model
        A two layer quadratic model.
    layer : int
        The layer to rotate.

    Returns
    -------
    Model
        The modified model.

    """
    
    # Extract low rank parts of quadratic kernel
    weights = model.get_weights()
    
    layer_index = 4 *(layer - 1)
    
    U = weights[layer_index][...,::2]
    V = weights[layer_index][...,1::2]
    
    shape = U.shape[:-1]
    num_feat = U.shape[-1]
    kernel_size = prod(shape)
    
    # Create a symmetric quadratic kernelfrom low rank parts
    U.shape = (kernel_size,num_feat)
    V.shape = (kernel_size,num_feat)
    
    J = dot(U,V.T)
    J = 0.5*(J+J.T)
    
    # Calculate eigenvalues and reduce to proper rank
    w,v = eigh(J)
    
    ind = abs(w).argsort()[-num_feat:]
    
    w = w[ind]
    v = v[:,ind]
    
    # Rotate suppressive features (those with negative eigenvalues)
    v.shape = shape + (num_feat,)
    
    v[...,w < 0] = rot90(v[...,w < 0],axes=(1,2))
    
    # Rewrite U and V as eigenvectors and eigenvectors times eigenvalues,
    # respetively
    weights[layer_index][...,::2]   = v
    weights[layer_index][...,1::2]  = v*w
    
    # Update the model's weights
    model.set_weights(weights)
    
    return model