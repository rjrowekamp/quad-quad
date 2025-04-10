# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:34:36 2017

@author: rrowekamp
"""

import os
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint,EarlyStopping
from numpy import array,corrcoef,zeros

from keras_models import make_model
from stim_tools import load_stim

def run_seed(spikes,
             stim,
             seed,
             job_params):
    """
    Fits the model with the data split into training and cross-validation 
    sets according to seed.

    Parameters
    ----------
    spikes : ndarray
        N x 1 array of spikes.
    stim : ndarray
        N x D_t x D_s1 x D_s2 x D_c array of stimuli (temporal dimenion, first
        spatial dimention, second spatial dimension, input channels)
    seed : int
        Seed to divide data by.
    job_params : dict
        Dictionary containing parameters specifying model and other things 
        needed to fit the model.

    Returns
    -------
    float
        Correlation between cross validation spikes and firing rate predicted 
        by model.

    """

    model_params = job_params['model_params']

    filename = f"{job_params['filename']}_seed-{seed}"

    weight_file = filename +'.weights.h5'

    print(weight_file)

    test_size   = job_params.get('test_size',0.25)
    patience    = job_params.get('patience',2)
    optimizer   = job_params.get('optimizer','adadelta')
    epochs      = job_params.get('epochs',1000)
    verbose     = job_params.get('verbose',2)

    stim_train,stim_valid,spikes_train,spikes_valid = train_test_split(stim,
        spikes,test_size=test_size,random_state=seed)

    callbacks = [EarlyStopping(patience=patience,min_delta=1e-5),
                 ModelCheckpoint(weight_file,save_weights_only=True,
                                 save_best_only=True)]

    model = make_model(**model_params)
    model.compile(optimizer=optimizer,loss='poisson')
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
    else:
        model.save_weights(weight_file)

    print('Inital error: training %.6g  validation %.6g' % 
          (model.evaluate(stim_train,spikes_train,verbose=0),
           model.evaluate(stim_valid,spikes_valid,verbose=0)))

    H = model.fit(stim_train,spikes_train,callbacks=callbacks,
                  validation_data=(stim_valid,spikes_valid),verbose=verbose,
                  epochs=epochs)

    array(H.history['loss']).tofile(filename + '.trainLoss')
    array(H.history['val_loss']).tofile(filename + '.validLoss')

    model.load_weights(weight_file)

    rate_valid = model.predict(stim_valid)[:,0]

    return corrcoef(rate_valid,spikes_valid)[0,1]

def make_model_params(stim_shape,
                      params,
                      modelType):
    
    L1 = params['L1'][0]
    L2 = params['L2'][0]

    D1 = int(params['D1'])
    D1 = int(D1)
    D2 = stim_shape[1] + 1 - D1

    modelParams = {}
    modelParams['in_layer'] = stim_shape
    l1p = {}
    l1p['kernel_shape'] = [1,D1,D1]
    if modelType[0] == 'Q':
        N1 = int(params['N1'])
        Q1 = params['Q1'][0]
        U1 = params['U1'][0]
        l1p['order'] = 2
        l1p['ranks'] = [1,N1]
        l1p['kregs'] = [{'type':'l2','lam2':10.**L1},
                        {'type':'Quad','Q1':10.**U1,'l2':10.**Q1}]
    else:
        l1p['order'] = 1
        l1p['ranks'] = [1,1]
        l1p['kregs'] = [{'type':'l2','lam2':10.**L1}]
    l1p['activation'] = 'sigmoid'
    l2p = {}
    l2p['kshape'] = [stim_shape[0],D2,D2]
    if modelType[1] == 'Q':
        N2 = int(params['N2'])
        Q2 = params['Q2'][0]
        U2 = params['U2'][0]
        l2p['order'] = 2
        l2p['ranks'] = [1,N2]
        l2p['kregs'] = [{'type':'l2','lam2':10.**L2},
                        {'type':'Quad','Q1':10.**U2,'l2':10.**Q2}]
    else:
        l2p['order'] = 1
        l2p['ranks'] = [1,1]
        l2p['kregs'] = [{'type':'l2','lam2':10.**L2}]
    l2p['activation'] = 'softplus'
    modelParams['layer_params'] = [l1p,l2p]

    return modelParams

def main(job_id : int,
         params : dict
         ) -> float:
    """
    Spearmint runs main with job_id and params. Fits four variations of the 
    model with different training/cross-validation splits

    Parameters
    ----------
    job_id : int
        This is the job_id-th suggestion for the hyperparameters from 
        Spearmint.
    params : dict
        Dictionary containing parameters related to the model and optimization.

    Returns
    -------
    float
        Negative mean correlation between cross-validation spikes and predicted 
        firing rate.

    """

    # Extrat the parameters for this specific job
    job_params = params['job_params']

    # LL, LQ, QL, or QQ, specifying whether each layer is linear or quadratic
    model_type = job_params['model']

    # Create beginning of the name for files associated with this run
    out_path    = job_params['out_path']
    dataset     = job_params['dataset']
    cell        = job_params['cell']
    tag         = f'model-{model_type}_run-{job_id}'
    
    job_params['filename'] = f'{out_path}{dataset}_{cell}_{tag}'

    # Load stimulus
    spikes,stim = load_stim(job_params)
    
    # Converts the hyperparameters associated with this run and the shape of 
    # the stimulus into paramters for keras_models
    model_params = make_model_params(stim.shape[1:],params,model_type)

    job_params['model_params'] = model_params

    # Fit four models with different divisions of data into train and cross-
    # validation sets and return the correlations with the cross-validation set
    corr = zeros(4)

    for seed in range(4):

        corr[seed] = run_seed(spikes,stim,seed,job_params)

    mean_corr = corr.mean()

    print(mean_corr)
    
    # Return negative correlation to make it a minimization problem
    return -mean_corr
