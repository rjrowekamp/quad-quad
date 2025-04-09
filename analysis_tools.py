#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:20:56 2024

@author: rrowekamp
"""

import pickle
from numpy import array,isnan,dot
import h5py

from spearmint_runner import make_model_params
from keras_models import make_model,Model

def get_best_run(file_name : str
                 ) -> tuple[int,float,dict]:
    """
    Takes the status file from Spearmint and finds the hyperparameters with 
    the lowest loss.

    Parameters
    ----------
    file_name : str
        Pickle file containing status.

    Returns
    -------
    tuple[int,float,dict]
        job_id with minimum loss, associate loss, hypeterparamters for that 
        job.

    """
    
    with open(file_name,'rb') as f:
        status = pickle.load(f)
        
    job_id  = array([job['id'] for job in status['jobs']])
    value   = array([job['values']['main'] for job in status['jobs']])
    
    job_id  = job_id[~isnan(value)]
    value   = value[~isnan(value)]
    
    ind = value.argmin()
    
    job_id  = job_id[ind]
    value   = value[ind]
    
    params = [job['params'] for job in status['jobs'] if job['id'] == job_id]
    
    for key in params.keys():
        
        params[key] = params[key]['values']
    
    return job_id,value,params

def make_best_model(status_file : str,
                    weight_file : str,
                    stim_shape  : tuple[int],
                    model_type  : str,
                    ) -> Model:
    """
    Takes status file and weight file and makes a model using the best 
    hyperpaarmeters.

    Parameters
    ----------
    status_file : str
        File containing status from Spearmint.
    weight_file : str
        File pointing to saved parameters with a %u where job_id should be.
    stim_shape : tuple[int]
        Shape of stimulus.
    model_type : str
        'QQ', 'LL', 'LQ', or 'LL'.

    Returns
    -------
    Model
        Model with lowest loss.

    """
    
        
    job_id,_,params = get_best_run(status_file)[2]
        
    model_params = make_model_params(stim_shape,params,model_type)
    
    model = make_model(**model_params)
    
    model.load_weights(weight_file % job_id)
        
    return model

def get_parameters(file_name : str,
                   model_type : str = 'QQ') -> dict:
    """
    Loads parameters from h5 file.

    Parameters
    ----------
    file_name : str
        The file to be loaded from.
    model_type : str, optional
        QQ, QL, LQ, or LL. If the first letter is Q, the first layer is 
        quadratic. If the second letter is Q, the second layer is quadratic. 
        The default is 'QQ'.

    Returns
    -------
    dict
        A dictionary with the parameters of the model.

    """
    
    param_file = h5py.File(file_name,'r')
    
    params  = {}
    params['A1'] = param_file['L1']['L1']['bias'][()]
    params['V1'] = param_file['L1']['L1']['kernel'][()]
    if model_type[0] == 'Q':
        params['UV1'] = param_file['Q1']['Q1']['kernel'][()]
        r1 = params['UV1'].shape[-1]/2
        params['J1'] = dot(params['UV1'][...,::2].reshape((-1,r1)),
                           params['UV1'][...,1::2].reshape((-1,r1)).T)
        params['J1'] = (params['J1'] + params['J1'].T)/2
    
    params['A2'] = param_file['L2']['L2']['bias'][()]
    params['V2'] = param_file['L2']['L2']['kernel'][()]
    params['UV2'] = param_file['Q2']['Q2']['kernel'][()]
    if model_type[1] == 'Q':
        r2 = params['UV2'].shape[-1]/2
        params['J2'] = dot(params['UV2'][...,::2].reshape((-1,r2)),
                           params['UV2'][...,1::2].reshape((-1,r2)).T)
        params['J2'] = (params['J2'] + params['J2'].T)/2
    
    params['D'] = param_file['Scale']['Scale']['kernel'][()]
    
    return params