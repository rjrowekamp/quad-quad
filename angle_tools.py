#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:41:56 2025

@author: rrowekamp
"""

from numpy import angle,exp,ndarray,ones,pi

def angle_mean(A            : ndarray,
               directional  : bool                      = False,
               axis         : int | tuple[int] | None   = None,
               weights      : ndarray | None            = None
               ) -> ndarray:
    """
    Calculates the weighted mean of an array of angles. Can be direction
    invariant or directional.

    Parameters
    ----------
    A : ndarray
        An array of angles to be averaged.
    directional : bool, optional
        Whether the angles are directions rather than orientations. The default 
        is False.
    axis : int | None, optional
        Whether to restrict the average to just some axes. The default is None.
    weights : ndarray | None, optional
        Weights for a weighted average. The default is None.

    Returns
    -------
    ndarray
        The mean angle.

    """
    
    if weights is None:
        
        weights = 1
    
    # If the angles are orientations (i.e. theta and theta + pi are equivalent),
    # factor doubles them so they are now offset by 2 pi and treated as the 
    # same. After the average, factor reduces the output to be between pi/2 and 
    # -pi/2.
    if directional:
        
        factor = 1
        
    else:
        
        factor = 2
        
    return angle((exp(1j*factor*A)*weights).mean(axis=axis))/factor

def angle_diff(A    : ndarray,
               B    : ndarray,
               directional  : bool  = False
               ) -> ndarray:
    """
    Calculates the absolute difference between two angles 

    Parameters
    ----------
    A : ndarray
        Array of angles.
    B : ndarray
        Another array of angles.
    directional : bool, optional
        Whether a shift of pi matters. The default is False.

    Returns
    -------
    ndarray
        The absolute difference in angles.

    """
    
    if directional:
        
        factor = 1
        
    else:
        
        factor = 2
        
    return (pi/factor) - abs((pi/factor) - abs(A-B) % (2*pi/factor))

def angle_std(A             : ndarray,
              directional   : bool                      = False,
              weights       : ndarray | None            = None,
              axis          : int | tuple[int] | None   = None
              ) -> ndarray:
    
    mean_A = anglemean(A,directional,weights,axis)
    
    if axis is not None:
        
        if isinstance(axis,int):
            axis = (axis,)
        
        new_shape = tuple([1 if j in axis else s for j,s in enumerate(A.hape)])
        
        mean_A.shape = new_shape
    
    if weights is None:
        
        weights = ones(1)
        
    return (angle_diff(A,mean_A,directional)*weights)