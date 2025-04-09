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
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from numpy import prod

# Sums the last dimension of its input
class SumLayer(Layer):
    """
    Sums the last dimension of the input.
    
    Attributes
    ----------
    kernel : The kernel is all ones, not trainable, and has no associated 
    bias.
    
    """

    def __init__(self, **kwargs) -> None:
        """
        The constructor of the SumLayer class.

        Parameters
        ----------
        **kwargs : TYPE
            The same arguments as the Layer class.

        Returns
        -------
        None.

        """
        super(SumLayer, self).__init__(**kwargs)

    def build(self, 
              input_shape : tuple) -> None:
        """
        Initializes the kernel as a vector of ones.

        Parameters
        ----------
        input_shape : tuple
            Shape of input. Used to determine size of kernel and shape of 
            the output.

        Returns
        -------
        None.

        """
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[-1], 1),
                                      initializer='ones',
                                      trainable=False)
        
        super(SumLayer, self).build(input_shape)

    def call(self, x):
        """
        Applies the layer to an input, summing the last dimension.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return K.dot(x, self.kernel)

    def compute_output_shape(self, 
                             input_shape : tuple) -> tuple:
        """
        Returns the output shape (same as the input shape except that the 
        last element is now 1).

        Parameters
        ----------
        input_shape : tuple[int]
            The shape of the input.

        Returns
        -------
        tuple[int]
            The shape of the output.

        """
        return input_shape[:-1]+(1,)

class SplitLayer(Layer):
    """
    Splits input into n parts along the last dimension.
    """

    def __init__(self, 
                 nsplit : int  = 2, 
                 **kwargs) -> None:
        """
        Constructor for SplitLayer class.

        Parameters
        ----------
        nsplit : int, optional
            The number of ways to split the output. The default is 2.
        **kwargs : TYPE
            Keyword arguments for the Layer class.

        Returns
        -------
        None

        """

        super(SplitLayer, self).__init__(**kwargs)

        self.nsplit = nsplit

    def call(self,x):
        """
        Splits x into several multiple outputs by taking every nth element 
        along the last dimension.

        Parameters
        ----------
        x : TYPE
            Input.

        Returns
        -------
        X : TYPE
            List of outputs.

        """

        KS = K.int_shape(x)

        n = KS[-1]//self.nsplit
        N = len(KS)

        X = [x[((N-1)*(slice(None),)+(slice(j,None,self.nsplit),))] for j in range(self.nsplit)]

        for x in X:
            x._keras_shape = KS[:-1]+(n,)

        return X

    def compute_output_shape(self,
                             input_shape : tuple
                             ) -> tuple:
        """
        Returns output shape. Same as input shape except that the last 
        value is divided by nsplit.

        Parameters
        ----------
        input_shape : tuple[int]
            Shape of input.

        Returns
        -------
        tuple[int]
            Shape of output.

        """

        output_shape = input_shape[:-1] + (input_shape[-1]//self.nsplit,)

        return self.nsplit*[output_shape]

    def get_config(self):
        
        config = {'nsplit': self.nsplit}
        base_config = super(SplitLayer, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):

        return self.nsplit*[None]
    
# Encourages low rank quadratic term's components to be identical up to a negative sign
class QuadRegularizer(Regularizer):
    """
    Regularizer for quadratic kernel that adds option to push U and V to 
    cover the same space.
    
    """

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

            UU = [tf.tensordot(U[:,j,:],U[:,j,:],([1],[1])) for j in range(self.ranks[1])]
            VV = [tf.tensordot(V[:,j,:],V[:,j,:],([1],[1])) for j in range(self.ranks[1])]
            DD = [uu-vv for uu,vv in zip(UU,VV)]

            if self.Q1 > 0.:
                reg += self.Q1*sum([K.sum(K.abs(dd)) for dd in DD])

            if self.Q2 > 0.:
                reg += self.Q1*sum([K.sum(K.square(dd)) for dd in DD])

        return reg

    
def make_model(in_layer     : tuple | Tensor,
               layer_params : dict,
               scale_layer  : bool = True,
               use_names    : bool = True,
               last_layer   = None
               ) -> Model:
    """
    Constructs a linear and/or quadratic model according to the given 
    parameters

    Parameters
    ----------
    in_layer : tuple[int] | Tensor
        The shape of the input or an input layer.
    layer_params : list[dict]
        List of dictionaries defining each layer of the model.
        
        Each dictionary consists of:
            kernel_shape : tuple[int]
                Shape of this layer's kernel.
            order : int
                Whether the model is linear (1) or quadratic (2).
            ranks : list[int]
                The first value is the number of units in this layer. The 
                second value determines the rank of the quadratic term, if 
                present.
            kernel_regs : list[dict[str:str,str:float]] | None
                'type' from 'l1','l2','l1l2','Quad' to determine which 
                regularizer to use
                strings and floats specify what values are used 
                for regularization
            activation : str
                Activation function of the layer
            kernel_initialization : None | str
                If 'uniform', weights are sampled uniformly
    scale_layer : bool, optional
        Whether the output of the final layer will be multiplied by a 
        trainable scale parameter. The default is True.
    use_names : bool, optional
        Whether to specify the names of the layers. The default is True.
    last_layer : Tensor | None, optional
        The layer feeding into the model if different from in_layer. The 
        default is None.

    Returns
    -------
    Model
        Keras model specified by input.

    """

    if isinstance(in_layer,tuple):
        in_layer = Input(in_layer)
        last_layer = in_layer
    else:
        if last_layer is None:
            last_layer = in_layer
            
    if use_names:
        layer_num = 0
    else:
        layer_num = None
        
    for lp in layer_params:
        if use_names:
            layer_num += 1
        last_layer = make_layer(last_layer,layer_num=layer_num,**lp)

    last_layer = Flatten()(last_layer)
        
    if scale_layer:
        if use_names:
            name = 'Scale'
        else:
            name = None
        last_layer = Dense(1,use_bias=False,kernel_initializer='ones',
                          name=name)(last_layer)

    return Model(in_layer,last_layer)

def make_layer(in_layer     : Tensor,
               kernel_shape : tuple,
               order        : int,
               ranks        : list,
               kregs        : list,
               activation   : str | None = None,
               kernel_init  : str | None = None,
               layer_num    : int | None = None
               ) -> Tensor:
    """
    Adds a linear or quadratic layer 

    Parameters
    ----------
    in_layer : Tensor
        Layer feeding into this layer.
    kernel_shape : tuple[int]
        The shape of the kernel of this layer.
    order : int
        The order of the layer: 1 (linear) or 2 (quadratic).
    ranks : list[int]
        The first value is the number of units in this layer. If order is 2,
        the second value determines the rank of the quadratic term.
    kregs : list[dict]
        Specifications for the regularization of the first and second order 
        terms.
    activation : str | None, optional
        Name of the activation layer. The default is None.
    kernel_init : str | None, optional
        Uniform initialization if 'uniform'. The default is None.
    layer_num : int | None, optional
        The layer number used in layer name. No name specified if None. The 
        default is None.

    Returns
    -------
    Tensor
        Output tensor of layer.

    """

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
                    IS = K.int_shape(in_layer)
                    kreg += [QuadRegularizer(ranks,kernel_shape+IS[-1:],**k)]
                else:
                    kreg += [None]
            else:
                kreg += [None]

    in_shape = tuple(in_layer.shape)[1:-1]
    
    out_shape = tuple([a - b + 1 for a,b in zip(in_shape,kernel_shape)])

    LP = {}

    if kernel_init == 'uniform':
        val = (6./(prod(kernel_shape)+1.))**0.5
        LP['kernel_initializer'] = initializers.RandomUniform(minval=-val,maxval=val)

    if layer_num is not None:
        name = 'L%i' % layer_num
    else:
        name = None

    linear = Conv3D(ranks[0],kernel_shape,kernel_regularizer=kreg[0],name=name,
                    **LP)(in_layer)

    if order == 2:
        if layer_num is not None:
            name = 'Q%i' % layer_num
        else:
            name = None

        QP = {}
        if kernel_init == 'uniform':
            val = (3./(prod(kernel_shape)+2.*ranks[0]*ranks[1]))**0.5
            QP['kernel_initializer'] = initializers.RandomUniform(minval=-val,
                                                                  maxval=val)
        quadA = Conv3D(2*ranks[1]*ranks[0],kernel_shape,
                       kernel_regularizer=kreg[1],use_bias=False,name=name,
                       **QP)(in_layer)
        quadU,quadV = SplitLayer()(quadA)
        quadC = Multiply()([quadU,quadV])
        quadD = Reshape(out_shape+(ranks[0],ranks[1]))(quadC)
        quadE = Dense(1,trainable=False,use_bias=False,kernel_initializer='ones')(quadD)
        quad = Reshape(out_shape+(ranks[0],))(quadE)
        layer = Add()([linear,quad])
    else:
        layer = linear

    if activation is not None:
        return Activation(activation)(layer)
    else:
        return layer

