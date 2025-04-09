#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:26:03 2024

@author: rrowekamp
"""

from numpy import mgrid,exp,concatenate,pi,zeros,ones,dot,sqrt,cos,sin,ndarray,array
from numpy.linalg import norm,eigh
from typing import Tuple

from diff_evol import DEFunction

class QuadraticGaborFunc(DEFunction):
    """
    Tries to fit the matrix J as the weighted sum of two-dimensional Gabor 
    wavelets.
    
    Attributes
    ----------    
    one_sided : int
       Whether to fit using only positive weights (>0), only negative weights 
       (<0), or both (=0).
    num_feat : int
        The number of Gabor wavelets or pairs of wavelets to fit
    J : ndarray
        The matrix to be fit.
    D : int
        The edge size of the Gabor wavelets.
    x, y : ndarray
        Arrays containing the position of each pixel in the D x D space.
    gabor_params : int
        The number of parameters required to specify each Gabor wavelet or pair 
        of wavelets and the associated weight.
    J0 : float
        The error relative to a zero matrix. Used to normalize the error.
    curved : bool
        Whether the Gabor wavelets have a curvature parameter.
    paired : bool
        Whether the Gabor wavelets each have their own weight and a phase 
        parameter or if they come in pairs with a shared weight and shared 
        parameters except for their phases, which are 0 degrees and 90 degrees.
    num_params : int
        The total number of parameters of the model. num_feat * gabor_params.
    addtype : ndarray
        Array specifying the type of addition used for each parameter.
    pmin, pmax : ndarray
        Arrays containing the minimum/maximum value for each parameter.
    """

    def __init__(self,
                 J          : ndarray,
                 num_feat   : int,
                 one_sided  : int   = 0,
                 curved     : bool  = False,
                 paired     : bool  = False,
                 bound      : bool  = True) -> None:
        """
        The constructor for the QuadraticGaborFunc class.

        Parameters
        ----------
        J : ndarray
            Matrix to be fit.
        num_feat : int
            The number of weighted features to used to fit J.
        one_sided : int, optional
            Whether to fit using only positive weights (>0), only negative weights 
            (<0), or both (=0). The default is 0.
        curved : bool, optional
            Whether the Gabors are curved. The default is False.
        paired : bool, optional
            Whether a Gabor wavelet feature consists of a sinlge Gabor wavelet 
            or as a quadrature pair (wavelets with the same parameters except 
            one has a phase of 0 degrees and the other has a phase of 90 degrees.
            The default is False.
        bound : bool, optional
            Whether parameters are required to stay between pmin and pmax. 
            Setting this to false may create issues with aliasing. The default 
            is True.

        Returns
        -------
        None
        """

        self.one_sided = one_sided
        self.num_feat = num_feat
        
        self.J = J.copy()
        self.D = int(self.J.size**0.25)
        self.J.shape = (self.D**2,self.D**2)
        self.J = 0.5*(self.J+self.J.T)
        
        self.x,self.y = mgrid[:self.D,:self.D]
        self.x.shape = (-1,1)
        self.y.shape = (-1,1)
        self.gabor_params = 8 + curved - paired
        self.J0 = (J**2).sum()
        self.curved = curved
        self.paired = paired
        self.bound = bound
        
        self.num_params = self.gabor_params * self.num_feat
        
        self.addtype,self.pmin,self.pmax = self.params_init()

    def make_gabor(self,
                   P : ndarray) -> Tuple[ndarray,ndarray]:
        """
        

        Parameters
        ----------
        P : ndarray
            A set of parameters to generate the corresponding Gabor wavelets 
            and weights from.

        Returns
        -------
        tuple[ndarray,ndarray]
            Returns an array with the Gabor wavelets and an array with their 
            corresponding weights.
        """

        P = P.copy().reshape((self.gabor_params,-1))

        W,X,Y,T,S,G,L = P[:7,:]
        
        if not self.paired:
            
            H = P[7,:]
            
        if self.curved:
            
            K = P[:,-1]

        Z = (self.x-X+self.y*1j-Y*1j)*exp(1j*T)
        U,V = Z.real,Z.imag
        
        if self.curved:
            
            U = U - K*V**2

        G = exp(-(U**2+V**2*G**2)/2./S**2)*exp(2*pi*1j/(S*L)*U)
        
        if self.paired:
            
            G = concatenate([G.real,G.imag],axis=1)
            W = concatenate([W,W])
            
        else:
            
            G = (G*exp(H*1j)).real
        
        G /= norm(G,axis=0)

        return G,W

    def make_J(self,
               P : ndarray) -> ndarray:
        """
        Construct the estimated J matrix for a set of parameters.

        Parameters
        ----------
        P : ndarray
            A set of parameters specifying the Gabor wavelets used to fit J.

        Returns
        -------
        ndarray
            Returns the approximation of J specified by P.

        """

        G,W = self.make_gabor(P)

        return dot(G*W,G.T)

    def eval(self,
             P : ndarray) -> float:
        """
        Calculates the error between J and the matrix specified by P.

        Parameters
        ----------
        P : ndarray
            A set of paratemers.

        Returns
        -------
        float
            The error of the fit, normalized so that a zero matrix has an 
            error of 0.

        """

        return ((self.J-self.make_J(P))**2).sum()/self.J0

    def params_init(self) -> Tuple[ndarray,ndarray,ndarray]:
        """
        Generates arrays the addition type, minimum value, and maximum value 
        associated with each parameter.

        Returns
        -------
        tuple[ndarray,ndarray,ndarray]
            Arrays specifying the addition type, minimum value, and maximum 
            value.
        """

        pmin = zeros((self.gabor_params,self.num_feat))
        pmax = zeros((self.gabor_params,self.num_feat))
        addtype = zeros((self.gabor_params,self.num_feat))

        if self.one_sided > 0:
            wmax = 1.1*eigh(self.J)[0][-1]
            pmin[0,:] = 0.
            pmax[0,:] = wmax
        elif self.one_sided < 0:
            wmax = 1.1*eigh(self.J)[0][0]
            pmin[0,:] = wmax
            pmax[0,:] = 0.
        else:
            wmax = 1.1*abs(eigh(self.J)[0]).max()
            pmin[0,:] = -wmax
            pmax[0,:] = wmax

        # X0 and Y0 (center)
        pmin[1:3,:] = 0.
        pmax[1:3,:] = self.D-1

        # Theta (orientation)
        pmin[3,:] = -pi
        pmax[3,:] = pi
        addtype[3,:] = 2

        # Sigma (controls size of envelope)
        pmin[4,:] = 4.
        pmax[4,:] = self.D
        addtype[4,:] = 1

        # Gamma (controls aspect ratio)
        pmin[5,:] = 2./3.
        pmax[5,:] = 3./2.
        addtype[5,:] = 1

        # Lambda (spatial wavelength)
        pmin[6,:] = 1.
        pmax[6,:] = 3.
        addtype[6,:] = 1
        
        if not self.paired:

            # Phi (Gabor phase)
            pmin[7,:] = -pi
            pmax[7,:] = pi
            addtype[7,:] = 2
            
        if self.curved:
            
            pmin[-1,:] = -2.
            pmax[-1,:] = 2.

        return addtype.flatten(),pmin.flatten(),pmax.flatten()

class LinearGaborFunc(DEFunction):
    """
    Tries to fit a vector as a two-dimensional Gabor wavelet.
    
    Attributes
    ----------  
    G : ndarray
        Vector to be fit.
    D : int
        Number of pixels per side of the Gabor wavelet.
    z : ndarray
        The x and y coordinates as a complex number.
    bound : bool
        Whether parameters outside of pmin and pmax are allowed.
    num_params: int
        The number of parameters defining the Gabor wavelet.
    addtype : ndarray
        Array specifying the type of addition used for each parameter.
    pmin, pmax : ndarray
        Arrays containing the minimum/maximum value for each parameter.       
    """

    def __init__(self,
                 G      : ndarray,
                 bound  : bool = True
                 ) -> None:
        """
        Constructor for LinearGaborFunc.

        Parameters
        ----------
        G : ndarray
            The feature to be fit as a two-dimensional Gabor wavelet.
        bound : bool, optional
            Whether to constrain parametres to initial range. The default is 
            True.

        Returns
        -------
        None
        """

        self.G = G.copy().flatten()/norm(G)
        self.D = sqrt(self.G.size)
        x,y = mgrid[:self.D,:self.D]
        self.z = (x+1j*y).flatten()
         
        self.bound = bound
         
        self.num_params = 7
         
        self.addtype,self.pmin,self.pmax = self.params_init()

    def eval(self,
             params : ndarray) -> float:
        """
        Measure the quality of the fit of a set of parameters using one minus 
        the normalized dot product.

        Parameters
        ----------
        params : ndarray
            Set of parameters to be evaluated.

        Returns
        -------
        float
            The loss value.
        """

        G = self.makeG(params)

        return 1-(G*self.G).sum()


    def make_gabor(self,
                   params : ndarray) -> ndarray:
        """
        Generates a two-dimensional Gabor wavelet from a set of parameters.

        Parameters
        ----------
        params : ndarray
            Parameters to use.

        Returns
        -------
        ndarray
            The corresponding Gabor wavelet.

        """

        x,y,t,s,a,l,h = params

        g = self.z.copy()
        g = g - (x+y*1j)
        g *= exp(1j*t)
        g = exp(-(g.real**2+g.imag**2*a**2)/2/s**2)*exp(1j*(2*pi/l*g.real+h))
        g = g.real
        g /= norm(g)

        return g

    def params_init(self) -> Tuple[ndarray,ndarray,ndarray]:
        """
        Creates array definining the properties of each parameter in a set.

        Returns
        -------
        tuple[ndarray,ndarray,ndarray]
            Addtype (parameter addition type), pmin (minimum values), and 
            pmax (maximum values).

        """

        addtype = zeros(self.num_params)
        pmin = zeros(self.num_params)
        pmax = zeros(self.num_params)

        addtype[:2] = 0
        pmin[:2] = 0.5
        pmax[:2] = self.D - 1.5

        pmin[2] = -pi
        pmax[2] = pi
        addtype[2] = 2

        pmin[3] = 2.
        pmax[3] = self.D/4.
        addtype[3] = 1

        pmin[4] = 2./3.
        pmax[4] = 3./2.
        addtype[4] = 1

        pmin[5] = 4.
        pmax[5] = self.D/2.
        addtype[5] = 1

        pmin[6] = -pi
        pmax[6] = pi
        addtype[6] = 2

        return addtype.flatten(),pmin.flatten(),pmax.flatten()
    
class QuadraticMovingSineFunc(DEFunction):
    """
    Tries to fit a J matrix as a combination of moving sine gratings.
    
    Attributes
    ----------
    one_sided : int
        Whether to fit only positive weights, only negative weigths, or both 
        at once
    num_feat : int
        The number of features to uste.
    J : ndarray
        The matrix to be fit.
    num_time : int
        The size of the features in the time dimension.
    num_color : int
        The number of colors to fit (1 for gray).
    D : int
        The size of the features in the vertical and horizontal dimensions.
    t,x,y : ndarray
        Mesh grids used to generate features.
    sine_params : int
        The number of parameters for each feature.
    J0 : float
        Error of a zero matrix.
    bound : bool
        Whether parameters are bound within initial range.
    num_params : int
        The total number of parameters in a parameter set.
    addtype : ndarray
        Type of addition for each parameter.
    pmin, pmax : ndarray
        Minimum/maximum value for each parameter.
    """

    def __init__(self,
                 J          : ndarray,
                 num_feat   : int,
                 num_time   : int,
                 one_sided  : int = 0,
                 num_color  : int = 1,
                 bound      : bool = True) -> None:
        """
        Constructor for QuadraticMovingSineFunc

        Parameters
        ----------
        J : ndarray
            Matrix to be fit.
        num_feat : int
            Number of features to use to fit J.
        num_time : int
            Size of time dimension.
        one_sided : int, optional
            Whether to fit only positive weights (>0), only negative weights 
            (<0), or both at once (=0). The default is 0.
        num_color : int, optional
            Number of colors to fit. The default is 1, which is gray.
        bound : bool, optional
            Whether parameters are bound within initial range. The default is
            True.

        Returns
        -------
        None
        """

        self.one_sided = one_sided
        self.num_feat = num_feat
        self.J = J.copy()
        self.num_time = num_time
        self.num_color = num_color
        self.D = int((J.size/(num_time*num_color)**2)**0.25)
        self.t,c,self.x,self.y,a = mgrid[:num_time,:1,:self.D,:self.D,:1]
        self.sine_params = 5 + num_time
        if self.num_color > 1:
            self.sine_params += self.num_color
        self.J0 = (self.J**2).sum()
        self.bound = bound
        
        self.num_params = self.sine_params*self.num_feat
        
        self.addtype,self.pmin,self.pmax = self.params_init()

    def make_sine(self, 
                  P : ndarray) -> Tuple[ndarray,ndarray]:
        """
        Makes moving sine gratings along with their corresponding weights.

        Parameters
        ----------
        P : ndarray
            Set of parameters defining the features and their weights.

        Returns
        -------
        tuple[ndarray,ndarray]
            Two-dimensional array with moving sine gratings. One-dimensional 
            array of corresponding weights.

        """

        P = P.copy().reshape((self.sine_params,self.num_feat))

        W,T,O,L,H = P[:5,:]
        if self.num_color > 1:
            C = P[5:5+self.num_color,:].reshape((self.num_color,1,1,self.num_feat))
            C  = C / norm(C,axis=0)
        else:
            C = ones(1)
        A = P[-self.num_time:,:].reshape((self.num_time,1,1,1,self.num_feat))
        A = A / norm(A,axis=0)

        z = self.x*cos(T) + self.y*sin(T)

        G = cos(2*pi/L*z+H+O*self.t)*A*C

        G.shape = (-1,W.size)

        G /= norm(G,axis=0)

        return G,W

    def make_J(self,
               P : ndarray) -> ndarray:
        """
        Makes moving sine gratings and combines them into a matrix using the 
        given weights.

        Parameters
        ----------
        P : ndarray
            Set of parameters to use.

        Returns
        -------
        ndarray
            The approximation of the matrix J based on the parameters.

        """

        G,W = self.make_sine(P)

        return dot(G*W,G.T)

    def eval(self,
             P : ndarray) -> float:
        """
        Returns normalized squared loss between J and the approximation given 
        by the parameters.

        Parameters
        ----------
        P : ndarray
            A set of parameters.

        Returns
        -------
        float
            The loss associated with the given set of parameters.

        """

        return ((self.J-self.make_J(P))**2).sum()/self.J0

    def params_init(self) -> Tuple[ndarray,ndarray,ndarray]:
        """
        Specifies the properties of each parameter in a set.

        Returns
        -------
        tuple[ndarray,ndarray,ndarray]
            Returns addtype (the type of addition each parameter uses), pmin 
            (the minimum value of each parameter), and pmax (the maximum value
            for each parameter).

        """

        pmin = zeros((self.sine_param,self.num_feat))
        pmax = zeros((self.sine_param,self.num_feat))
        addtype = zeros((self.sine_param,self.num_feat))

        if self.one_sided > 0:
            wmax = 2.1*eigh(self.J)[0][-1]
            pmin[0,:] = 0.
            pmax[0,:] = wmax
        elif self.one_sided < 0:
            wmax = 2.1*eigh(self.J)[0][0]
            pmin[0,:] = wmax
            pmax[0,:] = 0.
        else:
            wmax = 2.1*abs(eigh(self.J)[0]).max()
            pmin[0,:] = -wmax
            pmax[0,:] = wmax

        # Orientation
        pmin[1,:] = -pi
        pmax[1,:] = pi
        addtype[1,:] = 2

        # Spatial wavelength
        pmin[2,:] = 4.
        pmax[2,:] = self.D*2.
        addtype[2,:] = 1

        # Angular velocity
        pmin[3,:] = -pi
        pmax[3,:] = pi
        addtype[3,:] = 2

        # Initial phase
        pmin[4,:] = -pi
        pmax[4,:] = pi
        addtype[4,:] = 2

        # Color weight
        if self.num_color > 1:
            pmin[5:5+self.num_color,:] = -1
            pmax[5:5+self.num_color,:] = 1

        # Time weight
        pmin[-self.num_time:,:] = 0.
        pmax[-self.num_time:,:] = 1

        return addtype.flatten(),pmin.flatten(),pmax.flatten()

class MaxMotionFunc(DEFunction):
    """
    Finds the moving sine grating that maximizes/minimizes the output of a 
    model.
    
    Attributes
    ----------
    model : keras.Model
        The model being characterized.
    t,x,y : ndarray
        Mesh grids to create stimulus.
    use_color : bool
        Whether to make a color stimulus or a gray stimulus.
    num_params : int
        The number of parameters defining the stimulus.
    sign : int
        If searching for maximum resposne, this multiplies the otuput by -1 to 
        make it a minimization problem.
    amplitude : float
        The amplitude of the grating.
    bound : bool
        Whether parameters are constrained to be within initial values.
    addtype : ndarray
        Addition type for each parameter.
    pmin, pmax : ndarray
        Minimum and maximum values for each parameter.
    
    """
    
    def __init__(self,
                 model,
                 stim_shape     : Tuple[int,...],
                 use_color      : bool,
                 do_max         : bool,
                 amplitude      : float = 1.,
                 bound          : bool  = True) -> None:
        """
        Constructor for MaxMotionFunc.

        Parameters
        ----------
        model : keras.Model
            Model being characterized.
        stim_shape : tuplep[int]
            Shape of model input.
        use_color : bool
            Whether to makea  colored stimulus.
        do_max : bool
            Whether to maximize the stimulus.
        amplitude : float
            The amplitude of the grating. The default is 1.
        bound : bool, optional
            Whether parameters are bound by pmin and pmax. The default is True.

        Returns
        -------
        None
        """
        
        self.model = model
        t,x,y = mgrid[:stim_shape[0],:stim_shape[1],:stim_shape[2]]
        self.t = t/stim_shape[0]
        self.x = x/stim_shape[1]
        self.y = y/stim_shape[2]
        self.use_color = use_color
        if self.use_color:
            self.num_params = 7
        else:
            self.num_params = 4
        if do_max:
            self.sign = -1
        else:
            self.sign = 1
            
        self.amplitude = amplitude
            
        self.bound = bound
            
        self.addtype,self.pmin,self.pmax = self.params_init()
            
    def params_init(self) -> Tuple[ndarray,ndarray,ndarray]:
        """
        Returns arrays specifying properties of each parameter.

        Returns
        -------
        tuple[ndarray,ndarray,ndarray]
            Addtype (type of addition), pmin (minimum value), and pmax 
            (maximum value).

        """
        
        addtype = 2*ones(self.num_params)
        pmin = -pi*ones(self.num_params)
        pmax = pi*ones(self.num_params)
        if self.use_color:
            addtype[-2:] = 0
            pmax[-2:] = 1
            pmin[-2:] = -1
            
        return addtype.flatten(),pmin.flatten(),pmax.flatten()
    
    def fromHSV(self, hsv : ndarray) -> ndarray:
        """
        Converts HSV values into RGB.

        Parameters
        ----------
        hsv : ndarray
            HSV parameters.

        Returns
        -------
        ndarray
            Corresponding RGB color.

        """
        
        c = hsv[1]*hsv[2]
        h = hsv[0]*3/pi
        x = c*(1-abs(h % 2 - 1))
        m = hsv[2] - c
        
        if h < -2:
            return array([0,x,c])+m
        if h < -1:
            return array([x,0,c])+m
        if h < 0:
            return array([c,0,x])+m
        if h < 1:
            return array([c,x,0])+m
        if h < 2:
            return array([x,c,0])+m
        else:
            return array([0,c,x])+m
        
    def make_stim(self, params: ndarray) -> ndarray:
        """
        Makes a stimulus from the parameters.

        Parameters
        ----------
        params : ndarray
            Set of parameters for the stimulus.

        Returns
        -------
        ndarray
            The stimulus.

        """
        
        stim = self.amplitude*cos(params[0]*self.t+params[1]*self.x+params[2]*
                                  self.y+params[3])
        
        if self.use_color:
            rgb = self.fromHSV(params[4:])
            stim = stim[:,None,...]*rgb[:,None,None]
            stim.shape = (1,-1)+stim.shape[2:]+(1,)
        else:
            stim.shape = (1,)+stim.shape+(1,)
            
        return stim
    
    def eval(self,params : ndarray) -> float:
        """
        Evaluates response to stimulus defined by params.

        Parameters
        ----------
        params : ndarray
            Set of parameters to be evaluated.

        Returns
        -------
        float
            The response of the model/negative response of the model.

        """
        
        return self.sign*self.model.predict(self.make_stim(params))[0,0]
    