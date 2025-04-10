# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:30:08 2025

@author: Ryan

Placemarker file for the functions that load the data to be analyzed so other 
files can reference it.
"""

def load_stim(job_params):
    
    """
    Loads spikes and stimulus as specified in job_params
    
    Paramters
    ---------
    job_params : dict
        Dictionary with elements specifying what data will be analyzed by the 
        model
        
    Returns
    -------
    spikes : ndarray
        N x 1 array of responses associated with stim
    stim : ndarray
        N x D_t x D_s1 x D_s2 x D_c array of stimuli (temporal dimenion, first
        spatial dimention, second spatial dimension, input channels)
        
    Raises
    ------
    NotImplementedError
        This function must be written for each dataset to load the associated 
        data.
    
    """
    
    raise NotImplementedError()
    
    return spikes, stim