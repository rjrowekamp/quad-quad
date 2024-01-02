# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:34:36 2017

@author: rrowekamp
"""

import os
from sys import stdout
import Names
import stimXML as XML
from sklearn.model_selection import train_test_split
from KerasModels import PrepStim,makeModel
from keras.callbacks import ModelCheckpoint,EarlyStopping
from numpy import array,corrcoef,zeros

def loadStim(name,job_params):
    nlags = job_params.get('nlags',5)
    delay = job_params.get('delay',0)
    colorInvariant = job_params.get('colorInvariant',True)

    Y,S = XML.loadV4MFastColor(name+'.xml',delay=delay,nlags=1,xmlpath='/home/rrowekamp/V4/Data/')[:2]

    S,US,SS = XML.normStim(S)

    S = PrepStim(S,nlags=nlags,colorInvariant=colorInvariant)
    Y = Y[nlags-1:]

    stdout.write('Loaded data\n')
    stdout.flush()

    return Y,S

def runSeed(Y,S,seed,job_params):

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

    model = makeModel(S.shape[1:],**modelParams)
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

def makeModelParams(params,modelType,Init=None,nlags=5):

    params1 = {}
    for k in params.keys():
        params1[k] = params[k]['values']

    params = params1

    L1 = params['L1'][0]
    L2 = params['L2'][0]

    D1 = int(params['D1'])
    D1 = int(D1)
    D2 = 21-D1

    modelParams = {}
    modelParams['layerType'] = 2
    l1p = {}
    if Init is not None:
        l1p['Init'] = Init
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
    if Init is not None:
        l2p['Init'] = Init
    l2p['kshape'] = [nlags*3,D2,D2]
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

    jobnum = job_params['cell']

    zoomnum = (jobnum - 1) % 2
    cellnum = (jobnum - 1) // 2

    name = Names.V4Names()[cellnum] + ['_full','_half'][zoomnum]
    stdout.write(name+'\n')
    stdout.flush()

    L1 = params['L1'][0]
    L2 = params['L2'][0]

    D1 = int(params['D1'])
    D1 = int(D1)
    D2 = 21-D1

    modelParams = {}
    modelParams['layerType'] = 2
    l1p = {}
    if 'Init' in job_params:
        l1p['Init'] = job_params['Init']
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
    if 'Init' in job_params:
        l2p['Init'] = job_params['Init']
    l2p['kshape'] = [job_params.get('nlags',5)*3,D2,D2]
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

    job_params['modelParams'] = modelParams
    ptag = job_params.get('FileTag','')

    tag = name + ptag + '_model-%s_run-%u' % (modelType,job_id)

    JPATH = '/home/rrowekamp/V4/MJ/Results/Spear/Jobs/Results_%u/' % jobnum
    job_params['FileName'] = JPATH + tag + '_seed-%u'

    Y,S = loadStim(name,job_params)

    CV = zeros(4)

    for seed in range(4):

        CV[seed] = runSeed(Y,S,seed,job_params)

    cvm = CV.mean()

    print(cvm)

    return -cvm
