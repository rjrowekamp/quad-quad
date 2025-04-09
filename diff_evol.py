# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:22:03 2015

@author: rrowekamp
"""
from numpy import zeros,exp,log,angle,arange,inf,concatenate,ndarray
from numpy import all as numpy_all
from numpy.random import default_rng
from pickle import dump
import copy
from enum import Flag,auto
from typing import Tuple,Optional,Union

class DEFunction():
    """
    Base class for loss functions used by this module.
    
    Attributes
    ----------
    num_params : int
        The number of parameters in a parameter set.
    addtpe : ndarray
        An array indicating what type of addition should be used for each 
        parameter. 0: c = a + b. 1: c = exp(log(a)+log(b)),
        2: c = angle(exp(1j*(a+b))).
    pmin : ndarray
        An array containing the minimum value of each parameter
    pmax : ndarray
        An array containing the maximum value of each parameter
    bound : bool
        Whether values outside the bounds of pmin and pmax should be rejected.
    """

    def __init__(self) -> None:
        
        raise NotImplementedError()
        
    def eval(self,
             params
             ) -> float:
        
        raise NotImplementedError()

    # Evalutes cost for each parameter set
    def _eval(self,
             params : ndarray,
             error  : ndarray
             ) -> None:
        """
        Puts the loss of the parameters in params into the array error.

        Parameters
        ----------
        params : ndarray
            Two-dimensional array of parameter sets to be evaluated.
        error : ndarray
            One-dimensional array to receive the error values.

        Returns
        -------
        None
        """
        
        if self.bound:
                    
            # Set all elements to infinite.
            error[:] = inf
        
            # Find which sets of parameters are in bounds.
            ind = numpy_all(params >= self.pmin,axis=1)
            ind = ind & numpy_all(params <=self.pmax,axis=1)
        
            # Only evaluate the error for sets that are in bounds.
            error[ind] = [self.eval(p) for p in params[ind,:]]
            
        else:
            
            error[:] = [self.eval(p) for p in params]
            
    # Gives pmin, pmax, and addtype
    def paramsInit(self) -> Tuple[ndarray,ndarray,ndarray]:
        
        raise NotImplementedError()
        

class DEParams():
    """
    Handles population of parameters for differential evolution, including 
    mutation.
    
    Attributes
    ----------
    params : ndarray
        Parameter values. Shape (num_groups,num_params).
    num_groups : int
        The number of sets of parameters.
    num_params : int
        The number of parameters in one set.
    CR : ndarray
        Crossover mutation rate for each parameter set. Shape (num_groups,).
    FL : float
        Step-size lower bound.
    FU : float
        Step-size lower bound.
    F : ndarray
        Step-size for each parameter set. Shape (num_groups,).
    func : DEFunction
        Loss function to optimize.
    error : ndarray
        Current loss. Shape (num_groups,).
    mutate_unique : bool
        Whether to ensure unique sets are used for mutation (slower).
    init_gen : default_rng
        Generator used to initialize parameters.
    mutate_gen : default_rng
        Generator used for mutation.
    tau_C : float
        Chance of an element of CR mutating.
    tau_F : float
        Chance of an element of F mutating.
    num_diff : int
        Number of differences to add to parameter when mutating.
    """

    def __init__(self,
                 func           : DEFunction,
                 num_groups     : Optional[int] = None,
                 FL             : float         = 0.1,
                 FU             : float         = 0.9,
                 init_seed      = None,
                 mutate_seed    = None,
                 mutate_unique  : bool          = False,
                 tau_C          : float         = 0.1,
                 tau_F          : float         = 0.1,
                 num_diff       : int           = 2
                 ) -> None:
        """
        The constructor for the DEParams class.

        Parameters
        ----------
        func : DEFunction
            Loss function to optimize.
        num_groups : int | None, optional
            Number of sets of parameters to create. The default is None, which 
            sets num_groups to 10*num_params.
        FL : float, optional
            Lower limit on F. The default is 0.1.
        FU : float, optional
            Upper limit on F. The default is 0.9.
        init_seed : TYPE, optional
            Seed used to randomly initialize parameters. The default is None.
        mutate_seed : TYPE, optional
            Seed used to determine random mutations. The default is None.
        mutate_unique : bool, optional
            Whether whether sets used to mutate parameters are unique. This is 
            slower than allowing repeated sets. The default is False.
        tau_C : float, optional
            Chance of an element of CR mutating.
        tau_F : float, optional
            Chance of an element of F mutating.
        num_diff : int, optional
            Number of differences in parameter sets used during mutation.

        Returns
        -------
        None
        """
        
        self.func = func
        self.num_params = self.func.num_params

        if num_groups is None:
            self.num_groups = self.num_params*10
        else:
            self.num_groups = num_groups

        self.params = zeros((self.num_groups,self.num_params))
        self.CR = zeros((self.num_groups,))
        self.F = zeros((self.num_groups,))
        self.FL = FL
        self.FU = FU
        self.error = zeros((self.num_groups,))
        self.init_gen = default_rng(init_seed)
        self.mutate_gen = default_rng(mutate_seed)
        self.mutate_unique = mutate_unique
        self.tau_C = tau_C
        self.tau_F = tau_F
        self.num_diff = num_diff
        
        self.randomize_params()
        self.eval()
        
    def __copy__(self):
        
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        new.params  = self.params.copy()
        new.error   = self.error.copy()
        
        return new

    def randomize_params(self) -> None:
        """
        Randomly generates parameters according to the associated function.

        Returns
        -------
        None
        """

        RS = self.init_gen
         
        # Linear and angular variables are generated uniformly
        ind02 = (self.func.addtype == 0)+( self.func.addtype == 2)
         
        # Log variables are generated log uniformly
        ind1 = self.func.addtype == 1
         
        delta02 = self.func.pmax[ind02]-self.func.pmin[ind02]
        delta1 = log(self.func.pmax[ind1])-log(self.func.pmin[ind1])
         
        self.params[:,ind02] = delta02*RS.uniform(size=(self.num_groups,ind02.sum()))
        self.params[:,ind02] += self.func.pmin[ind02]
        self.params[:,ind1] = delta1*RS.uniform(size=(self.num_groups,ind1.sum())) 
        self.params[:,ind1] += log(self.func.pmin[ind1])
        self.params[:,ind1] = exp(self.params[:,ind1])

        # Mutation rates uniformly distributed from 0 to 1
        self.CR = RS.uniform(size=self.num_groups)
         
        # Scale parameters uniformly distributed from FL to FL+FU
        self.F = self.FL + self.FU*RS.uniform(size=self.num_groups)

    def mutate(self) -> "DEParams":
        """
        Creates a new population based on the current population.
        
        Returns
        -------
        child : DEParams
            A new population of parameters generated from this population..
        """
        
        # Initialize object to hold new parameters
        child = copy.copy(self)

        # Get indices of different addtypes
        ind0 = self.func.addtype == 0 # Linear
        ind1 = self.func.addtype == 1 # Log
        ind2 = self.func.addtype == 2 # Angular

        # Generate new parameters
        RS = self.mutate_gen
        
        # Pick members to be used to generate new parameters
        if self.mutate_unique:
            # This ensures uniqueness at the cost of speed
            M = concatenate([RS.choice(self.num_groups-1,replace=False,size=(1,2*self.num_diff+1)) for _ in range(self.num_groups)])
        else:
            # This trades uniqueness for speed
            M = RS.integers(self.num_groups-1,size=(self.num_groups,2*self.num_diff+1))
        
        # Ensure the member being mutated isn't included
        M[M > arange(self.num_groups)[:,None]] += 1
        
        # Mutate CR and F first with given probability
        ind_CR = RS.uniform(size=self.num_groups) < self.tau_C
        child.CR[ind_CR] = RS.uniform(size=ind_CR.sum())
        ind_F = RS.uniform(size=self.num_groups) < self.tau_F
        child.F[ind_F] = child.FL + child.FU * RS.uniform(size=ind_F.sum())
        
        # Determine which paramters to mutate with rate CR
        mask = RS.uniform(size=self.params.shape) < child.CR[:,None]
        
        ind_G = arange(self.num_groups)
        
        # Ensure at least one parameter changes
        mask[ind_G,RS.integers(self.num_params,size=self.num_groups)] = True
        
        # Create copy of params and move into linear space
        params0 = self.params[:,ind0].copy()
        params1 = log(self.params[:,ind1].copy())
        params2 = exp(1j*self.params[:,ind2].copy())
        
        # Divide mask
        mask0 = mask[:,ind0]
        mask1 = mask[:,ind1]
        mask2 = mask[:,ind2]
        
        # Create children
        params0 = (1-mask0)*params0 + mask0*params0[M[:,0],:] + mask0*child.F[:,None]*(params0[M[:,1::2],:].sum(1)-params0[M[:,2::2],:].sum(1))
        params1 = (1-mask1)*params1 + mask1*params1[M[:,0],:] + mask1*child.F[:,None]*(params1[M[:,1::2],:].sum(1)-params1[M[:,2::2],:].sum(1))
        params2 = (1-mask2)*params2 + mask2*params2[M[:,0],:] + mask2*child.F[:,None]*(params2[M[:,1::2],:].sum(1)-params2[M[:,2::2],:].sum(1))
        
        # Copy to child
        child.params[:,ind0] = params0
        child.params[:,ind1] = exp(params1)
        child.params[:,ind2] = angle(params2)

        # Calculate costs for new parameter sets
        child.eval()

        return child

    def eval(self) -> None:
        """
        Updates error according to the current parameter values.

        Returns
        -------
        None
        """
        
        self.func._eval(self.params,self.error)

    def merge(self,
              child : Optional["DEParams"] = None
              ) -> "DEParams":
        """
        Merges this DEParams with another and returns the better set of 
        parameters in each parameter slot. Generates a new DEParams object 
        using mutate if none is given.

        Parameters
        ----------
        child : DEParams | None, optional
            DEParams object to merge with. If None, a new DEParams object is 
            created using mutate. The default is None.

        Returns
        -------
        new : DEParams
            The merged DEParams object.
        """

        # Create new object
        new = copy.copy(self)

        # Create child if not given one
        if child is None:
            child = self.mutate()

        # Select groups where each is better
        ind1 = self.error > child.error
        ind2 = True ^ ind1

        # Put best sets into new object
        new.params[ind1,:] = child.params[ind1,:]
        new.params[ind2,:] = self.params[ind2,:]
        new.error[ind1] = child.error[ind1]
        new.error[ind2] = self.error[ind2]
        new.CR[ind1] = child.CR[ind1]
        new.CR[ind2] = self.CR[ind2]
        new.F[ind1] = child.F[ind1]
        new.F[ind2] = self.F[ind2]

        return new

    def params_min(self) -> ndarray:
        """
        Returns the set of parameters with the lowest error.

        Returns
        -------
        ndarray
            The best set of parameters in the population.
        """

        return self.params[self.error.argmin(),:].copy()

class Converged(Flag):
    """
    Flag to indicate whether the trainer has converged and if so how.
    """
    NOT_CONVERGED   = 0
    IT_MAX          = auto()
    MIN_DELTA       = auto()
    TARGET_ERROR    = auto()

class DETrainer():
    """
    Takes a DEFunction and DEParams and optimizes them until convergence.
    
    Attributes
    ----------
    func : DEFunction
        The function being optimized
    num_groups : int | None
        The number of sets of parameters to include in a population. If None,
        DEParams will use its default.
    converged : Converged
        Whether a covergence criterion has been reached. Use of Converged Flag 
        allows one to see which convergence criteria were met.
    its : int
        The number of iterations that have been started.
    DEP : DEParams
        The DEParams object containing the current population.
    e1 : float
        The current mean error of the population.
    e0 : float | None
        The mean error from the last iteration. Is None if its is zero.
    """

    def __init__(self,
                 func,
                 DEP        : DEParams = None,
                 num_groups : int = None
                 ) -> None:
        """
        The constructor for the DETrainer class.

        Parameters
        ----------
        func : DEFunction
            Function to be optimized.
        DEP : None | DEParams, optional
            Starting population of parameter sets. The default is None.
        num_groups : int, optional
            Number of sets of parameters in the population. The default is None.

        Returns
        -------
        None.

        """

        self.func = func
        
        self.num_groups = num_groups

        self.converged = Converged.NOT_CONVERGED
        self.its = 0

        if DEP is None:
            self.DEP = DEParams(self.func,self.num_groups)
            self.e1 = self.DEP.error.mean()
        else:
            self.DEP = DEP
            self.e1 = self.DEP.error.mean()
        self.e0 = None

    def train(self,
              num_its       : Union[int,float]  = inf,
              reset         : bool              = False,
              verbose       : bool              = False,
              mindelta      : float             = 0.,
              save_file     : Optional[str]     = None,
              save_freq     : Optional[int]     = 1,
              tag           : str               = '',
              target_error  : float             = -inf
              ) -> None:
        """
        Generates new populations through DEParams' merge function until 
        convergence criteria are met.

        Parameters
        ----------
        num_its : int | float, optional
            Number of additional iterations to run before terminating. The 
            default is inf.
        reset : bool, optional
            Whether to overwrite existing DEP and start from scratch. The 
            default is False.
        verbose : bool, optional
            Whether to output text showing optimization progress. The default 
            is False.
        mindelta : float, optional
            Minimum fractional change in mean error to continue optimizing. 
            The default is 0.
        save_file : str | None, optional
            Filename to save intermediate state of self for recovery or 
            monitoring progress. The default is None.
        save_freq : int | None, optional
            How many iterations between saving to saveFile. This has no effect 
            if saveFile is not defined. The default is 1.
        tag : str, optional
            A tag to be added to the printed output if verbose is enabled. 
            The default is ''.
        target_error : float | None, optional
            A value of the minimum error that will end optimization. The 
            default is None.

        Returns
        -------
        None.
        """
        
        
        if reset:
            # Return to initial state
            self.DEP = DEParams(self.func,self.num_groups)
            self.e1 = self.DEP.error.mean()
            self.its = 0
            self.e0 = None

        itmax = self.its + num_its
        
        if save_file is not None :
            
            with open(save_file,'wb') as f:
                dump(self,f)
        
        self.converged = Converged.NOT_CONVERGED

        while not self.converged :
            
            self.its += 1
            
            self.e0 = self.e1
            self.DEP = self.DEP.merge()
            self.e1 = self.DEP.error.mean()
            emin = self.DEP.error.min()
            
            if verbose:
                print(f'{tag} {self.its} {self.e1:.15f} {emin:.15f}')
                
            # Check convergence criteria. Value of converged will indicate 
            # which criteria caused termination (if any)
            if self.its >= itmax:
                # If maximum number of iterations have been reached
                self.converged |= Converged.IT_MAX
            if self.e1 - self.e0 >= -mindelta*self.e0:
                # If error did decrease more than minimum
                self.converged |= Converged.MIN_DELTA
            if emin < target_error:
                # If minimum error has reache target
                self.converged |= Converged.TARGET_ERROR
                
            if save_file is not None:
                if (self.its % save_freq) == 0:
                    try:
                        with open(save_file,'wb') as f:
                            dump(self,f)
                    except KeyboardInterrupt:
                        # If KeryboardInterupt occurs during file output, try
                        # saving the file again to avoid leaving an empty file.
                        print("Finishing saving file.")
                        with open(save_file,'wb') as f:
                            dump(self,f)
                        raise

    def params_min(self) -> ndarray:
        """
        Returns the set of parameters with the minimum error.

        Returns
        -------
        ndarray
            The set of parameters.
        """

        return self.DEP.params_min()


