#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:42:02 2024

@author: rrowekamp
"""

from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Activation,Add,Dense,Flatten,Input,Multiply
from keras.layers.convolutional import Conv3D
from keras.models import Model
from numpy import cos,exp,mgrid,ndarray,pi,save,sin,zeros
from numpy.linalg import norm
from numpy.random import poisson
from sklearn.model_selection import train_test_split

import keras_models as KM

def make_model() -> Model:
    """
    Makes an empty two-layer quadratic model to be modified later.

    Returns
    -------
    Model
        The model.

    """
    
    input_layer     = Input((5,20,20,1),name='Input')

    linear_layer1   = Conv3D(1,(1,10,10),name='Linear1')(input_layer)
    
    vector_layer1   = Conv3D(8,(1,10,10),use_bias=False,name='Quad1')(input_layer)
    mult_layer1     = Multiply()(KM.SplitLayer()(vector_layer1))
    quad_layer1     = Dense(1,use_bias=False,kernel_initializer='ones',trainable=False,name='Ones1')(mult_layer1)
    
    layer1          = Activation('sigmoid')(Add()([linear_layer1,quad_layer1]))
    
    linear_layer2   = Conv3D(1,(5,11,11),name='Linear2')(layer1)
    
    vector_layer2   = Conv3D(8,(5,11,11),use_bias=False,name='Quad2')(layer1)
    mult_layer2     = Multiply()(KM.SplitLayer()(vector_layer2))
    quad_layer2     = Dense(1,use_bias=False,kernel_initializer='ones',trainable=False,name='Ones2')(mult_layer2)
    
    layer2          = Activation('softplus')(Add()([linear_layer2,quad_layer2]))
    
    out_layer       = Dense(1,use_bias=False,name='Output',kernel_initializer='ones')(Flatten()(layer2))
    
    model           = Model(input_layer,out_layer)
    
    return model

def make_motion_model() -> Model:
    """
    Makes a model selective for a particular moving grating.

    Returns
    -------
    Model
        The motion-selective model.

    """
    
    model = make_model()
    
    weights = model.get_weights()
    
    x1,y1 = mgrid[:10,:10]
    
    q1_g1 = exp(-((x1-4.5)**2+(y1-6.5)**2)/2/2**2)*sin(2*pi/8*(x1-4.5))
    q1_g1 /= norm(q1_g1)
    weights[0][0,...,0,0] = q1_g1
    weights[0][0,...,0,1] = q1_g1
    
    q1_g2 = exp(-((x1-4.5)**2+(y1-2.5)**2)/2/2**2)*sin(2*pi/8*(x1-4.5))
    q1_g2 /= norm(q1_g2)
    weights[0][0,...,0,2] = q1_g2
    weights[0][0,...,0,3] = q1_g2
    
    q1_g3 = exp(-((x1-4.5)**2+(y1-6.5)**2)/2/2**2)*sin(2*pi/8*(y1-4.5))
    q1_g3 /= norm(q1_g3)
    weights[0][0,...,0,4] = q1_g3
    weights[0][0,...,0,5] = -q1_g3
    
    q1_g4 = exp(-((x1-4.5)**2+(y1-2.5)**2)/2/2**2)*sin(2*pi/8*(y1-4.5))
    q1_g4 /= norm(q1_g4)
    weights[0][0,...,0,6] = q1_g4
    weights[0][0,...,0,7] = -q1_g4
    
    l1 = exp(-((x1-4.5)**2+(y1-4.5)**2)/2/2**2)*sin(2*np.pi/8*(y1-4.5))
    l1 /= norm(l1)
    weights[1][0,...,0,0] = l1
    
    t2,x2,y2 = mgrid[:5,:11,:11]
    
    q2_g1 = exp(-t2/5)*sin(2*pi*(x2/8.+t2/5))
    q2_g1 /= norm(q2_g1)
    weights[4][...,0,0] = q2_g1
    weights[4][...,0,1] = q2_g1
    
    q2_g2 = exp(-t2/5)*sin(2*pi*(x2/8.+t2/5+0.25))
    q2_g2 /= norm(q2_g2)
    weights[4][...,0,2] = q2_g2
    weights[4][...,0,3] = q2_g2
    
    q2_g3 = exp(-t2/5)*sin(2*pi*(x2/8.-t2/5))
    q2_g3 /= norm(q2_g3)
    weights[4][...,0,4] = q2_g3
    weights[4][...,0,5] = -q2_g3
    
    q2_g4 = exp(-t2/5)*sin(2*pi*(x2/8.-t2/5+0.25))
    q2_g4 /= norm(q2_g4)
    weights[4][...,0,6] = q2_g4
    weights[4][...,0,7] = -q2_g4

    l2 = exp(-t2/5)*cos(2*pi*(x2-5)/8)
    l2 /= norm(l2)
    weights[5][...,0,0] = l2
    
    weights[2] *= 0
    weights[6] *= 0
    weights[8][...] = 1
    
    model.set_weights(weights)
    
    return model

def make_spikes(stim    : ndarray
                ) -> ndarray:
    """
    Generates spikes using a poisson distribution from the rate given by the 
    motion model on the given stimulus.

    Parameters
    ----------
    stim : ndarray
        The stimulus to generate spikes from.

    Returns
    -------
    ndarray
        The spike response to the stimulus.

    """
    
    model = make_motion_model()
    
    r = model.predict(stim)
    
    # Normalizing to ensure all stimuli produce the same firing rate and can 
    # be more fairly compared
    r /= r.mean()
    
    return poisson(r)
    
def fit_model(stim      : ndarray,
              spikes    : ndarray,
              DATA_PATH : str,
              filename  : str,
              seed      : int   = 0
              ) -> None:
    """
    Tries to reproduce the motion model using the given stimulus and spikes.

    Parameters
    ----------
    stim : ndarray
        The input of the model.
    spikes : ndarray
        The output of the model.
    DATA_PATH : str
        Where to store the outputs.
    filename : str
        A name to ditinguish this from other stimulus and spike pairs.
    seed : int, optional
        Used to randomly but reproducibly split the stimulus/spike pairs into 
        training and cross-validation sets. The default is 0.

    Returns
    -------
    None

    """
    
    S_t,S_v,Y_t,Y_v = train_test_split(stim,spikes,test_size=0.25,random_state=seed)

    weight_file = f'{DATA_PATH}Weights-{filename}_{seed}.weights.h5'
    
    callbacks = [EarlyStopping(patience=2,min_delta=1e-5),
                 ModelCheckpoint(weight_file,save_weights_only=True,save_best_only=True)]

    model = make_model()
    model.compile('adadelta','poisson')
    r = model.predict(S_t)
    weights = model.get_weights()
    weights[-1] *= Y_t.mean()/r.mean()
    model.set_weights(weights)

    H = model.fit(S_t,Y_t,callbacks=callbacks,validation_data=(S_v,Y_v),verbose=2,epochs=10000)

    save(f'{DATA_PATH}Fitting/Loss-{filename}_{seed}.npy',H.history['loss'])
    save(f'{DATA_PATH}Fitting/ValidLoss-{filename}_{seed}.npy',H.history['val_loss'])
    
def load_model(DATA_PATH    : str,
               filename     : str,
               seed         : int   = 0
               ) -> Model:
    """
    Load a model fit previously.

    Parameters
    ----------
    DATA_PATH : str
        The path to where the resutls were stored.
    filename : str
        The name used to distinguish this model from others.
    seed : int, optional
        The seed used to divide training and cross-validation data. The default is 0.

    Returns
    -------
    Model
        The model.

    """
    
    weight_file = f'{DATA_PATH}Weights-{filename}_{seed}.weights.h5'
    
    model = make_model()
    
    model.load_weights(weight_file)
    
    return model

def generate_reduced_stimulus(nphi  : int   = 20
                              ) -> ndarray:
    """
    Generates a bunnch of different moving gratings to test response of model.

    Parameters
    ----------
    nphi : int, optional
        The number of orientations to test. The default is 20.

    Returns
    -------
    ndarray
        The group of moving gratings.

    """
    
    S = zeros((nphi,5,20,5,20,20))
    
    t,x,y =mgrid[:5,:20,:20]
    
    for phi in range(nphi):
        for tt in range(5):
            for nk in range(20):
                S[phi,tt,nk,...] = cos(2*pi*(nk*x/20+tt*t/5+phi/nphi))
                    
    return S.reshape((-1,5,20,20,1))
    
def test_model(DATA_PATH    : str,
               filename     : str,
               seed         : int               = 0,
               S            : ndarray | None    = None
               ) -> ndarray:
    """
    Load model and calculate its response to a given stimulus.

    Parameters
    ----------
    DATA_PATH : str
        Where the model is stored.
    filename : str
        The naem of the model.
    seed : int, optional
        The seed used to split the data during training. The default is 0.
    S : ndarray | None, optional
        A stimulus to evaluate the model on. If None, the reduced grating 
        stimulus will be used. The default is None.

    Returns
    -------
    ndarray
        The response to the stimulus.

    """
    
    if S is None:
        S = generate_reduced_stimulus()
        
    model = load_model(DATA_PATH,filename,seed)
    
    return model.predict(S)[:,0]

