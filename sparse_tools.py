#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:27:56 2025

@author: rrowekamp
"""

from keras.models import Model
from numpy import float32,float64,ndarray,rot90

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

def rotate_layer1_sparse(model      : Model,
                         stimulus   : ndarray
                         ) -> float32 :
    """
    Calculates the sparsenss of the model if the first layer is rotated 90 
    degrees.

    Parameters
    ----------
    model : Model
        Model to be modified.
    stimulus : ndarray
        Stimulus to use to calculate responses.

    Returns
    -------
    float32
        .

    """
    
    weights = model.get_weights()
    
    weights[0] = rot90(weights[0],axes=(1,2))
    weights[2] = rot90(weights[2],axes=(1,2))
    
    model.set_weights(weights)
    
    return sparseness(model.predict(stimulus))