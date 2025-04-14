# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:38:35 2025

@author: Ryan
"""

from numpy import arange,arccosh,corrcoef,cosh,dot,inv,ones,permutation
from numpy import unique,zeros
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import MLencoding as MLE

class Extrap():
    
    def __init__(self,
                 rate,
                 spikes,
                 use_cosh = True):
        
        self.rate = rate
        self.spikes = spikes
        
        self.repeats = spikes.shape[1]
        
        if use_cosh:
            
            self.func = lambda x,y : arccosh(corrcoef(x,y)[0,1]**-2)
            self.ifunc = lambda x : cosh(x)**-0.5
            
        else:
            
            self.func = lambda x,y : corrcoef(x,y)[0,1]**-2
            self.ifunc = lambda x : x**-0.5
        
    def extrap(self,
               frac = 0.01*arange(30),
               n_rep = 100):
        
        n_drop = unique(int(frac*self.spikes.size))
        
        num_samples = zeros(n_rep*(n_drop > 0).sum()+(n_drop==0).sum())
        raw_corr = zeros(num_samples.size)
                
        for j,d in n_drop[n_drop > 0]:
            
            for k in range(n_rep):
                
                p = permutation(self.spikes.size)[:d]
                
                mask = ones(self.spikes)
                mask[p] == 0
                
                r_s = (self.spikes*mask).mean(1)
                
                num_samples[k+j*n_rep] = self.spikes.size-d
                raw_corr[k+j*n_rep] = self.func(r_s,self.rate)
                
        if 0 in n_drop:
            
            num_samples[-1] = self.spikes.size
            raw_corr[-1] = self.func(self.spikes.mean(1),self.rate)
            
        X = ones((num_samples.size,2))
        X[:,1] = 1/num_samples
        
        B = dot(inv(dot(X.T,X)),dot(X.T,raw_corr))
        
        return self.ifunc(B[0])
    
def train_xgboost(spikes,
                  stim,
                  n_pca = 100):
    
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