# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:15:37 2017

@author: rrowekamp
"""

from keras.models import Model
from keras.layers.convolutional import Conv3D
from keras.layers import Add,Multiply,Dense,Flatten,Input,Reshape
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation
from keras.regularizers import l1,l2,l1_l2,Regularizer
from keras import initializers
from numpy import prod

# Sums the last dimension of its input
class SumLayer(Layer):

    def __init__(self, **kwargs):
        super(SumLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[-1], 1),
                                      initializer='ones',
                                      trainable=False)
        super(SumLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]+(1,)

# Splits input into nsplit outputs by taking every nsplit-th element from the last dimension
class SplitLayer(Layer):

    def __init__(self, nsplit=2, **kwargs):

        super(SplitLayer, self).__init__(**kwargs)

        self.nsplit = nsplit

    def call(self,x):

        KS = K.int_shape(x)

        n = KS[-1]//self.nsplit
        N = len(KS)

        X = [x[((N-1)*(slice(None),)+(slice(j,None,self.nsplit),))] for j in range(self.nsplit)]

        for x in X:
            x._keras_shape = KS[:-1]+(n,)

        return X

    def compute_output_shape(self, input_shape):

        output_shape = input_shape[:-1] + (input_shape[-1]//self.nsplit,)

        return self.nsplit*[output_shape]

    def get_config(self):
        config = {'nsplit': self.nsplit}
        base_config = super(SplitLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):

        return self.nsplit*[None]
    
# Makes a multilayer linear and/or quadratic model based on the list of layerParams
def make_model(input_shape,layerParams,scaleLayer=True,inLayer=None,useNames=True,lastLayer=None):

    if inLayer is None:
        inLayer = Input(tuple(input_shape))
        lastLayer = inLayer
    else:
        if lastLayer is None:
            lastLayer = inLayer
            
    lastShape = input_shape[:-1]

    n = 0
    for lp in layerParams:
        n += 1
        lastLayer,lastShape = make_layer(lastLayer,lastShape,n=n,useNames=useNames,**lp)

    lastLayer = Flatten()(lastLayer)
        
    if scaleLayer:
        if useNames:
            name = 'Scale'
        else:
            name = None
        lastLayer = Dense(1,use_bias=False,kernel_initializer='ones',name=name)(lastLayer)


    return Model(inLayer,lastLayer)

# Encourages low rank quadratic term's components to be identical up to a negative sign
class QuadRegularizer(Regularizer):

    def __init__(self,ranks,kshape,l1=0.,l2=0.,Q1=0.,Q2=0.,type=None):

        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.Q1 = K.cast_to_floatx(Q1)
        self.Q2 = K.cast_to_floatx(Q2)
        self.ranks = ranks
        self.D = prod(kshape)

    def __call__(self,X):

        reg = 0.
        if self.l1 > 0.:
            reg += self.l1*K.sum(K.abs(X))

        if self.l2 > 0.:
            reg += self.l2*K.sum(K.square(X))

        if self.Q1 > 0. or self.Q2 > 0.:

            U = K.reshape(X[...,::2],(self.D,self.ranks[0],self.ranks[1]))
            V = K.reshape(X[...,1::2],(self.D,self.ranks[0],self.ranks[1]))

            UU = [K.dot(U[:,j,:],U[:,j,:].T) for j in range(self.ranks[0])]
            VV = [K.dot(V[:,j,:],V[:,j,:].T) for j in range(self.ranks[0])]
            DD = [uu-vv for uu,vv in zip(UU,VV)]

            if self.Q1 > 0.:
                reg += self.Q1*sum([K.sum(K.abs(dd)) for dd in DD])

            if self.Q2 > 0.:
                reg += self.Q1*sum([K.sum(K.square(dd)) for dd in DD])

        return reg

# Make linear or quadratic layer
def make_layer(inputLayer,input_shape,kshape,order,ranks,kregs,activation,n=None,Init=None,useNames=True):

    kshape = tuple(kshape)
    outshape = tuple([a-b+1 for a,b in zip(input_shape,kshape)])

    if kregs is None:
        kreg = order*[None]
    else:
        kreg = []
        for k in kregs:
            if isinstance(k,dict):
                if k['type'] == 'l1':
                    kreg += [l1(k['lam1'])]
                elif k['type'] == 'l2':
                    kreg += [l2(k['lam2'])]
                elif k['type'] == 'l1l2':
                    kreg += [l1_l2(k['lam1'],k['lam2'])]
                elif k['type'] == 'Quad':
                    IS = K.int_shape(inputLayer)
                    kreg += [QuadRegularizer(ranks,kshape+IS[-1:],**k)]
                else:
                    kreg += [None]
            else:
                kreg += [None]

    LP = {}

    if Init == 'uniform':
        val = (6./(prod(kshape)+1.))**0.5
        LP['kernel_initializer'] = initializers.RandomUniform(minval=-val,maxval=val)

    if n is not None and useNames:
        name = 'L%i' % n
    else:
        name = None

    linear = Conv3D(ranks[0],kshape,kernel_regularizer=kreg[0],name=name,**LP)(inputLayer)

    if order == 2:
        if n is not None and useNames:
            name = 'Q%i' % n
        else:
            name = None

        QP = {}
        if Init == 'uniform':
            val = (3./(prod(kshape)+2.*ranks[0]*ranks[1]))**0.5
            QP['kernel_initializer'] = initializers.RandomUniform(minval=-val,maxval=val)
        quadA = Conv3D(2*ranks[1]*ranks[0],kshape,kernel_regularizer=kreg[1],use_bias=False,name=name,**QP)(inputLayer)
        quadU,quadV = SplitLayer()(quadA)
        quadC = Multiply()([quadU,quadV])
        quadD = Reshape(outshape+(ranks[0],ranks[1]))(quadC)
        quadE = Dense(1,trainable=False,use_bias=False,kernel_initializer='ones')(quadD)
        quad = Reshape(outshape+(ranks[0],))(quadE)
        layer = Add()([linear,quad])
    else:
        layer = linear

    return Activation(activation)(layer),outshape

