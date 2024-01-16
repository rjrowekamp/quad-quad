# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:34:36 2017

@author: rrowekamp
"""

import os
from stim_tools import load_stim
from sklearn.model_selection import train_test_split
from keras_models import make_model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from numpy import array,corrcoef,zeros

def run_seed(Y,S,seed,job_params):

    modelParams = job_params['modelParams']

    FileName = job_params['FileName'] % seed

    weightFile = FileName +'.bestWeights'

    print(weightFile)

    test_size = job_params.get('test_size',0.25)
    patience = job_params.get('patience',2)
    optimizer = job_params.get('optimizer','adadelta')
    epochs = job_params.get('epochs',1000)

    S_train,S_valid,Y_train,Y_valid = train_test_split(S,Y,test_size=test_size,random_state=seed)

    callbacks = [EarlyStopping(patience=patience,min_delta=1e-5),ModelCheckpoint(weightFile,save_weights_only=True,save_best_only=True)]

    model = make_model(S.shape[1:],**modelParams)
    model.compile(optimizer=optimizer,loss='poisson')
    if os.path.exists(weightFile):
        model.load_weights(weightFile)
    else:
        model.save_weights(weightFile)

    print('Inital error: training %.6g  validation %.6g' % (model.evaluate(S_train,Y_train,verbose=0),model.evaluate(S_valid,Y_valid,verbose=0)))

    H = model.fit(S_train,Y_train,callbacks=callbacks,validation_data=(S_valid,Y_valid),verbose=2,epochs=epochs)

    array(H.history['loss']).tofile(FileName + '.trainLoss')
    array(H.history['val_loss']).tofile(FileName + '.validLoss')

    model.load_weights(weightFile)

    r_valid = model.predict(S_valid)[:,0]

    return corrcoef(r_valid,Y_valid)[0,1]

def makeModelParams(params,modelType,stim_shape):

    L1 = params['L1'][0]
    L2 = params['L2'][0]

    D1 = int(params['D1'])
    D1 = int(D1)
    D2 = stim_shape[1] + 1 - D1

    modelParams = {}
    l1p = {}
    l1p['kshape'] = [1,D1,D1]
    if modelType[0] == 'Q':
        N1 = int(params['N1'])
        Q1 = params['Q1'][0]
        U1 = params['U1'][0]
        l1p['order'] = 2
        l1p['ranks'] = [1,N1]
        l1p['kregs'] = [{'type':'l2','lam2':10.**L1},{'type':'Quad','Q1':10.**U1,'l2':10.**Q1}]
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
        l2p['kregs'] = [{'type':'l2','lam2':10.**L2},{'type':'Quad','Q1':10.**U2,'l2':10.**Q2}]
    else:
        l2p['order'] = 1
        l2p['ranks'] = [1,1]
        l2p['kregs'] = [{'type':'l2','lam2':10.**L2}]
    l2p['activation'] = 'softplus'
    modelParams['layerParams'] = [l1p,l2p]

    return modelParams

def main(job_id,params):

    job_params = params['job_params']

    modelType = job_params['model']

    dataset = job_params['dataset']
    cell = job_params['cell']
    
    spikes,stim = load_stim(dataset,cell,**job_params)
    
    model_params = makeModelParams(params,modelType,stim.shape[1:])

    job_params['modelParams'] = model_params
    
    ptag = job_params.get('FileTag','')

    tag = f'cell-{cell}{ptag}_model-{modelType}_run-{job_id}'

    out_path = f'../Results/dataset_{dataset}/'
    
    job_params['FileName'] = out_path + tag + '_seed-%u'

    spikes,stim = load_stim(dataset,cell,**job_params)

    corr = zeros(4)

    for seed in range(4):

        corr[seed] = run_seed(spikes,stim,seed,job_params)

    mean_corr = corr.mean()

    print(mean_corr)
    
    # Return negative correlation to make it a minimization problem
    return -mean_corr
