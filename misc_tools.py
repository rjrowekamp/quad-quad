#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:01:05 2025

@author: rrowekamp
"""

from numpy import diag,float64,log,ones,ndarray,sqrt,triu,zeros
from numpy.linalg import eigh
from numpy.random import default_rng,Generator

from diff_evol import DETrainer

def calc_AIC(rss            : float | float64,
             num_samples    : int,
             num_params     : int
             ) -> float64 :
    """
    Calcualtes AIC (up to a constant) for a mean squared error fit.

    Parameters
    ----------
    rss : float | float64
        The residual errors divided by num_samples.
    num_samples : int
        The number of samples being estimated.
    num_params : int
        The number of paramters used to estimate the samples.

    Returns
    -------
    float64
        The AIC.

    """
    
    return 2*num_params + num_samples*log(rss)

def calc_AIC_J(det1     : DETrainer,
               det2     : DETrainer | None  = None
               ) -> float64 :
    """
    Calcukate the AIC for a model fitting J.

    Parameters
    ----------
    det1 : DETrainer
        DETrainer to fit J.
    det2 : DETrainer | None
        Optional second DETrainer to consider, such as if the excitatory and 
        suppressive components were fit independently.

    Returns
    -------
    float64
        Akaike Information Criterion.

    """

    # The symmetric matrix beign fit
    J = det1.func.J

    param       = det1.params_min()
    num_params  = param.size
    J0          = det1.func.make_J(param)
    
    if det2 is not None:
        param       = det2.params_min()
        num_params  += param.size
        J0          += det2.func.make_J(param)
        
    # J is symmetric, so not all values are independent.
    num_samples = J.shape[0]*(J.shape[0]+1)//2
    
    rss = ((J-J0)**2).sum()/num_samples
    
    return calc_AIC(rss,num_samples,num_params)

def sig_dims_shuffle(J              : ndarray,
                     num_rep        : int   = 1000,
                     threshold      : float = 0.05,
                     seed                   = None,
                     subtract_mean  : bool  = False
                     ) -> tuple[ndarray] :
    """
    

    Parameters
    ----------
    J : ndarray
        A symmetric square matrix to be analyzed. The algorithm will try to 
        make the matrix square and force it to be symmetric.
    num_rep : int, optional
        Number of times the matrix should be shuffled. The default is 1000.
    threshold : float, optional
        The significance threshold. The default is 0.05.
    seed, optional
        A random generator or seed for a random generator. The default is None.
    subtract_mean : bool, optional
        Whether to subtract the mean from J before analyzeing. The default is 
        False.

    Returns
    -------
    tuple[ndarray]
        The sigificant eigenvalues and eigenvectors.

    """
    
    # Initialize random number generator if necessarily
    if isinstance(seed,Generator):
        
        rng = seed
        
    else:
        
        rng = default_rng(seed)
        
    # Make sure J is square and symmetric
    if J.ndim != 2:
        
        D = int(sqrt(J.size))
        J = J.reshape((D,D))
        
    J = (J+J.T)/2
    
    # Remove mean if told to do so
    if subtract_mean:
        
        J -= J.mean()
        
    # Calculate eigenvalues and eigenvecor
    values,vectors = eigh(J)
        
    #Create an index going from maximum absolute magnitude
    index = (-abs(values)).argsort()
    
    # Get elements along diagonal
    diag_J = diag(J)
    
    # Boolean index of upper triangle
    upper_triangle = triu(ones(J.shape),k=1) == 1
    
    # Get elements along upper triangle
    tri_J = J[upper_triangle]
    
    # Arrays to hold distributions of minimum and maximum eigenvalues
    max_eig = zeros(num_rep)
    min_eig = zeros(num_rep)
    
    # Dummy matrix to hold shuffled upper triangle
    upper_J = zeros(J.shape)
    
    for j in range(num_rep):
        
        # Shuffle diagonal and upper triangle seperately
        shuffle_diag_J  = rng.permutation(diag_J)
        shuffle_tri_J   = rng.permutation(tri_J)
        
        # Combine into a shuffled J matrix
        upper_J[upper_triangle] = shuffle_tri_J
        shuffle_J = diag(shuffle_diag_J) + upper_J + upper_J.T
        
        # Calculate the eigenvalues of the shuffled matrix
        shuffle_weights = eigh(shuffle_J)[0]
        
        # Store minimum and maximum eigenvalues
        max_eig[j] = shuffle_weights.max()
        min_eig[j] = shuffle_weights.min()
        
    # Start probability of non-significance to 0
    p = 0.
    
    num_sig = -1
    
    while p < threshold:
        
        num_sig += 1
        
        # Get next eigenvalue
        value = values[index[num_sig]]
        
        # Get probability of seeing an eigenvalue of that magnitude
        if value > 0:
            p0 = (value < max_eig).sum()/num_rep
        else:
            p0 = (value > min_eig).sum()/num_rep
            
        # Combine with previous probabilities
        p = 1 - (1-p) * (1-p0)
    
    # Extract significant eigenvalues and eigenvectors
    index = index[:num_sig]
    values = values[index]
    vectors = vectors[:,index]
    
    # Sort by eigenvalue
    index = values.argsort()
    
    return values[index], vectors[:,index]


        