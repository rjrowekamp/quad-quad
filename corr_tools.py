# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:38:35 2025

@author: Ryan
"""

from numpy import arange,arccosh,corrcoef,cosh,dot,inv,nan,nanmean,ndarray,ones
from numpy import permutation,unique,zeros
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import MLencoding as MLE

class Extrap():
    """
    Class to allow extrapolation of correlation to infinite data.
    
    Attributes
    ----------
    rate : ndarray
        The predicted firing rate to be evaluated. T elements in length.
    spikes : ndarray
        The observed responses to repeated stimuli. T x repeats in size.
    func : function
        The function to be linearly extrapolated with the inverse of the 
        number of samples.
    ifunc : function
        The inverse of func to convert the bias term into a correlation.
        
    """
    
    
    def __init__(self,
                 rate       : ndarray, # Prediction
                 spikes     : ndarray, # Observed responses to repeated stimuli
                 use_acosh  : bool = True): # Whether to compress correlation with arccosh
        
        self.rate = rate
        self.spikes = spikes
        
        self.repeats = spikes.shape[1]
        
        if use_acosh:
            
            self.func = lambda x,y : arccosh(corrcoef(x,y)[0,1]**-2)
            self.ifunc = lambda x : cosh(x)**-0.5
            
        else:
            
            self.func = lambda x,y : corrcoef(x,y)[0,1]**-2
            self.ifunc = lambda x : x**-0.5
        
    def extrap(self,
               frac     : ndarray   = 0.01*arange(30), # Fractions of responses to drop
               n_rep    : int       = 100): # Number of random samples to do for each fraction
        
        # Convert fractions into numbers of samples and drop duplicates
        n_drop = unique(int(frac*self.spikes.size))
        
        # Store number of samples used for each calculation and the correlation
        # For n_drop == 0, only one calculation is done because there is only one permutation
        num_samples = zeros(n_rep*(n_drop > 0).sum()+(n_drop==0).sum())
        raw_corr = zeros(num_samples.size)
                
        for j,d in enumerate(n_drop[n_drop > 0]):
            
            for k in range(n_rep):
                
                # Create mask
                p = permutation(self.spikes.size)[:d]
                mask = ones(self.spikes.shape)
                mask[p] == nan
                
                r_s = nanmean(self.spikes*mask,axis=1)
                
                num_samples[k+j*n_rep] = self.spikes.size-d
                raw_corr[k+j*n_rep] = self.func(r_s,self.rate)
        
        # Calculate full correlation for no dropped samples
        if 0 in n_drop:
            
            num_samples[-1] = self.spikes.size
            raw_corr[-1] = self.func(self.spikes.mean(1),self.rate)
        
        # Do linear regression
        X = ones((num_samples.size,2))
        X[:,1] = 1/num_samples
        
        B = dot(inv(dot(X.T,X)),dot(X.T,raw_corr))
        
        return self.ifunc(B[0])
    
def train_xgboost(spikes    : ndarray,
                  stim      : ndarray,
                  n_pca     : int   = 100):
    """

    Parameters
    ----------
    spikes : ndarray
        Observed responses to be predicted.
    stim : ndarray
        Stimuli to be used to predict spikes.
    n_pca : int, optional
        Number of pca dimensions to use. The default is 100.

    Returns
    -------
    models : TYPE
        A list of xgboost models fit with different training/validation splits.

    """
    
    models = []
    
    for seed in range(4):
        
        stim_train,stim_valid,spikes_train,spikes_valid = train_test_split(stim,
           spikes,test_size=0.25,random_state=seed)
        
        pca = PCA(n_pca)
        
        X_train = pca.fit_transform(stim_train)
        
        model = MLE.MLencoding(tunemodel='xgboost')
        
        model.fit(X_train,spikes_train)
        
        models.append(model)
        
    return models