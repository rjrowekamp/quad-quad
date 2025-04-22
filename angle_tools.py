#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:41:56 2025

@author: rrowekamp
"""

from numpy import angle,arctan,cos,dot,exp,mgrid,ndarray,ones,pi,sin
from numpy.fft import fftn
from numpy.linalg import eigh

from diff_evol import DETrainer

def angle_mean(A            : ndarray,
               directional  : bool                      = False,
               axis         : int | tuple[int] | None   = None,
               weights      : ndarray | None            = None
               ) -> ndarray:
    """
    Calculates the weighted mean of an array of angles. Can be direction
    invariant or directional.

    Parameters
    ----------
    A : ndarray
        An array of angles to be averaged.
    directional : bool, optional
        Whether the angles are directions rather than orientations. The default 
        is False.
    axis : int | None, optional
        Whether to restrict the average to just some axes. The default is None.
    weights : ndarray | None, optional
        Weights for a weighted average. The default is None.

    Returns
    -------
    ndarray
        The mean angle.

    """
    
    if weights is None:
        
        weights = 1
    
    # If the angles are orientations (i.e. theta and theta + pi are equivalent),
    # factor doubles them so they are now offset by 2 pi and treated as the 
    # same. After the average, factor reduces the output to be between pi/2 and 
    # -pi/2.
    if directional:
        
        factor = 1
        
    else:
        
        factor = 2
        
    return angle((exp(1j*factor*A)*weights).mean(axis=axis))/factor

def angle_diff(A    : ndarray,
               B    : ndarray,
               directional  : bool  = False
               ) -> ndarray:
    """
    Calculates the absolute difference between two angles 

    Parameters
    ----------
    A : ndarray
        Array of angles.
    B : ndarray
        Another array of angles.
    directional : bool, optional
        Whether a shift of pi matters. The default is False.

    Returns
    -------
    ndarray
        The absolute difference in angles.

    """
    
    if directional:
        
        factor = 1
        
    else:
        
        factor = 2
        
    return (pi/factor) - abs((pi/factor) - abs(A-B) % (2*pi/factor))

def angle_std(A             : ndarray,
              directional   : bool                      = False,
              weights       : ndarray | None            = None,
              axis          : int | tuple[int] | None   = None
              ) -> ndarray:
    """
    Calculates the angular analog of the standard devaition.

    Parameters
    ----------
    A : ndarray
        An array of angles to measure.
    directional : bool, optional
        Whether the angle is directional or direction-independent. The default is False.
    weights : ndarray | None, optional
        An array of weights. The default is None.
    axis : int | tuple[int] | None, optional
        The axes to calculate std on. The default is None.

    Returns
    -------
    ndarray
        Array of calculated standard devaitions.

    """
    
    mean_A = angle_mean(A,directional,weights,axis)
    
    if axis is not None:
        
        if isinstance(axis,int):
            axis = (axis,)
        
        new_shape = tuple([1 if j in axis else s for j,s in enumerate(A.hape)])
        
        mean_A.shape = new_shape
    
    if weights is None:
        
        weights = ones(1)
        
    if A.ndim > weights.ndim:
        
        weights.shape = (A.nidim-weights.ndim)*(1,) + weights.shape
        
    angle_std = angle_diff(A,mean_A,directional)**2
    angle_std = (angle_std*weights).sum(axis=axis)/weights.sum(axis=axis)
    
    return angle_std**0.5

def fft_orientation(A               : ndarray,
                    drop_corners    : bool  = False
                    ) -> ndarray:
    """
    Calculate dominant orientation of an array.

    Parameters
    ----------
    A : ndarray
        A two or more dimensional array to be analyzed.
    drop_corners : bool, optional
        Whether to drop values outside of a circle around zero frequency. The 
        default is False.

    Returns
    -------
    ndarray
        Mean orientation weighted by Fourier power.

    """

    # Calculate Fourier transform
    FA = fftn(A,axes=[-2,-1])

    # Zero out zero frequency
    FA[...,0,0] = 0

    D1,D2 = A.shape[-2:]

    # Zero out Nyqist frequency 
    if D1 % 2 == 0:
        FA[...,D1//2,:] = 0
    if D2 % 2 == 0:
        FA[...,D2//2] = 0

    # Get spatial frequency associated with each element
    x1,x2 = mgrid[:D1,:D2]

    k1 = 2*pi*x1/D1
    k2 = 2*pi*x2/D2

    k1 = (k1 + pi) % (2 * pi) - pi
    k2 = (k2 + pi) % (2 * pi) - pi

    # Only use points within a circle around zero frequency
    if drop_corners:

        k = k1**2 + k2**2

        FA[...,k > pi**2] = 0

    # Convert spatial frequencies to angles
    th = angle(k1 + 1j * k2)
    
    # Use fft power to calculate weighted average of angles
    return angle_mean(th,weights=abs(FA))

def fft_moving_vector(A             : ndarray,
                      drop_corners  : bool  = False
                      ) -> ndarray:
    """
    Calculate direction of motion using fft.

    Parameters
    ----------
    A : ndarray
        Three or more dimensional array to be analyzed.
    drop_corners : bool, optional
        Whether to drop corner values. The default is False.

    Returns
    -------
    ndarray
        The mean direction.

    """
    
    FA = fftn(A,axes=range(-3,0))

    FA[...,0,0] = 0
    FA[...,0,:,:] = 0

    DT,D1,D2 = A.shape[-3:]

    if DT % 2 == 0:
        FA[...,DT//2,:,:] = 0
    if D1 % 2 == 0:
        FA[...,D1//2,:] = 0
    if D2 % 2 == 0:
        FA[...,D2//2] = 0

    t1,x1,x2 = mgrid[:DT,:D1,:D2]

    k1 = ((2.*pi*x1/float(D1) + pi) % (2. * pi) - pi)
    k2 = ((2.*pi*x2/float(D2) + pi) % (2. * pi) - pi)
    w1 = ((2.*pi*t1/float(DT) + pi) % (2. * pi) - pi)

    if drop_corners:

        k = k1**2 + k2**2
        FA[...,k > pi**2] = 0

    k1 = k1 * w1
    k2 = k2 * w1

    th = angle(k1 + 1j * k2)

    TH = angle_mean(th,weights=abs(FA),directed=True)

    return TH

def shift_pi(th     : float | ndarray
             ) -> ndarray:
    """
    Shift angles to -pi to pi range.

    Parameters
    ----------
    th : float | ndarray
        Input angles in radians.

    Returns
    -------
    ndarray
        Output angles.

    """

    return (th + pi) % (2 * pi) - pi

def phase_diff_fft(A            : ndarray,
                   B            : ndarray,
                   axes         : int | tuple[int] | None   = None,
                   directional  : bool                      = False
                   ) -> ndarray:
    """
    Measure the mean phase difference between to arrays.

    Parameters
    ----------
    A,B : ndarray
        Arrays to be compared.
    axes : int | tuple[int] | None, optional
        Axes to be averaged over. The default is None.
    directional : bool, optional
        Whether orientation matters. The default is False.

    Returns
    -------
    ndarray
        Averaged phase differences.

    """

    #Calculate Fourier transforms
    FA = fftn(A,axes=axes)
    FB = fftn(B,axes=axes)

    # Calculate phase differences for each point
    DT = angle(FA) - angle(FB)

    # Return average phase difference weighted by multiplied magnitude of each 
    # element
    return angle_mean(DT,weights=abs(FA)*abs(FB),directional=directional)

def calc_ori_field(det          : DETrainer,
                   d_up         : int           = 3,
                   dot_weight   : bool          = True
                   ) -> tuple[ndarray,ndarray]:
    """
    

    Parameters
    ----------
    det : DETrainer
        DETrainer trained on QuadraticGaborFunction.
    d_up : int, optional
        The upsample factor. The default is 3.
    dot_weight : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    tuple[ndarray,ndarray]
        DESCRIPTION.

    """
    
    func    = det.func
    
    # Get size of filter
    D1  = func.x.max()+1
    D2  = func.y.max()+1

    # Create upsampled grid
    X,Y = mgrid[:(D1*d_up),:(D2*d_up)]
    X = X/d_up
    Y = Y/d_up
    X.shape = (-1,1)
    Y.shape = (-1,1)
    
    # Reshape params so each Gabor is a column
    num_params = func.num_params
    param    = det.paramsMin().reshape((num_params,-1))

    # Weight Gabors by their presence in the subspace covered by J
    if dot_weight:
        
        gabor    = func.make_gabor(param)[0]

        n    = func.num_feat

        J    = func.J
        
        # Get excitatory, suppressive, or combined subspace.
        if func.one_sided == 1:

            vJ = eigh(J)[1][:,-n:]
            
        elif func.one_sided == -1:
            
            vJ = eigh(J)[1][:,:n]
            
        else:
            
            wJ,vJ = eigh(J)
            vJ = vJ[:,abs(wJ).argsort()[-n:]]

        reduced_space   = dot(vJ,vJ.T)

        gabor_proj      = (dot(reduced_space,gabor)**2).sum(0)**0.5
        
    # Give Gabors equal weight
    else:
        
        gabor_proj = 1
    
    # Extract Gabor params
    w,x0,y0,th,sig,gam  = param[:6,:]
    
    # Calculate Gaussian profile
    z0  = x0 + y0*1j
    z   = (func.x + func.y*1j - z0)*exp(1j*th)
    x,y = z.real,z.imag
    mag = exp(-(x**2+gam**2*y**2)/2/sig**2)
    mag = (mag**2).sum(0)**0.5

    # Calculate the excitatory and suppressive orientations at each point
    s = sin(th)
    c = cos(th)

    if func.curved:
        
        kap     = param[-1,:]
        
        dUdX    = c-2*kap/sig*s*((Y-y0)*c+(X-x0)*s)
        dUdY    = -s-2*kap/sig*c*((Y-y0)*c+(X-x0)*s)
        
    else:
        
        dUdX    = c
        dUdY    = -s

    Z           = (X+Y*1j-z0)*exp(1j*th)
    XX,YY         = Z.real,Z.imag
    PROFILE     = exp(-(XX**2+gam**2*YY**2)/2/sig**2)*w*mag*gabor_proj
    
    TH  = arctan(dUdX/dUdY)+pi/2

    # Sum profile across Gabors
    profile     = abs(PROFILE).sum(1)
    ori         = angle_mean(TH,weights=abs(PROFILE),axis=1)

    return ori,profile

def calc_layer1_ex_sup_diff(det_ex      : DETrainer,
                            det_sup     : DETrainer,
                            d_up        : int   = 3,
                            dot_weight  : bool  = True
                            ) -> ndarray:
    """
    Calculate the difference in the mean excitatory and suppressive 
    orientations at their point of maximum interaction.

    Parameters
    ----------
    det_ex : DETrainer
        DETrainer trained on QuadraticGaborFunction for excitatory features.
    det_sup : DETrainer
        DETrainer tranid on QuadraticGaborfunction for suppressive features.
    d_up : int, optional
        The upsample factor. The default is 3.
    dot_weight : bool, optional
        Whether to weight Gabors by projections into subspace of J. The default 
        is True.

    Returns
    -------
    ndarray
        The orientation difference.

    """
    
    ori_ex,profile_ex       = calc_ori_field(det_ex,d_up,dot_weight)
    ori_sup,profile_sup     = calc_ori_field(det_sup,d_up,dot_weight)
    
    ind = (profile_ex*profile_sup).argmax()
    
    return angle_diff(ori_ex[ind],ori_sup[ind])