__author__ = 'Davide Lasagna, Politecnico di Torino Dipartimento di Ingegneria Aerospaziale. <davide.lasagna@polito.it>'
__date__ = '26/07/2011'
__licence_ = """
Copyright (C) 2011  Davide Lasagna

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__doc__ = """This module contains classes used to describe discrete time systems.

Classes
==================
.. currentmodule:: mpc.systems

.. autosummary::
   :toctree: generated
    
    mpc.systems.DtSystem
    mpc.systems.DtNLSystem
    mpc.systems.DtLTISystem
    mpc.systems.NoisyDtLTISystem
       

.. autoclass:: mpc.systems.DtSystem
    :members:
    :inherited-members:
    :show-inheritance:
    
.. autoclass:: mpc.systems.DtLTISystem
    :members: 
    :inherited-members:
    :show-inheritance:
    
.. autoclass:: mpc.systems.NoisyDtLTISystem
    :members: 
    :inherited-members:
    :show-inheritance:
    
.. autoclass:: mpc.systems.DtNLSystem
    :members:
    :inherited-members:
    :show-inheritance:
    
Exceptions
==========
.. autoclass:: mpc.systems.DtSystemError
.. autoclass:: mpc.systems.ObservabilityError
    
Functions
=========
.. autofunction:: mpc.systems.c2d
"""

import numpy as np
import scipy
import scipy.linalg


class DtSystem( object ):
    def __init__ ( self, n_states, n_inputs, n_outputs, Ts, x0 ):
        """Base class of all the linear/non-linear discrete time systems of
        the :py:mod:`mpc.systems` module. Use derived classes to work, 
        because this one is just here for the general structure.
        
        Parameters
        ----------
        n_states : int
            the length of the state column vector :math:`\\mathbf{x}(k)`
        n_inputs : int
            the number of system inputs, i.e., the length of the column 
            vector :math:`\\mathbf{u}(k)`
        n_outputs : int
            the number of system outputs, i.e., the length of the column 
            vector :math:`\\mathbf{y}(k)`    
        Ts : float
            the sampling interval
        x0 : np.matrix, with shape ``(n_states, 1)``
            the initial system state
            
        Attributes
        ----------
        x : np.matrix object with shape ``(n_states, 1)``
            the state of the system at time step :math:`k`
        n_states : int
            the length of the state column vector :math:`\\mathbf{x}(k)`
        n_inputs : int
            the number of system inputs, i.e., the length of the column 
            vector :math:`\\mathbf{u}(k)`
        n_outputs : int
            the number of system outputs, i.e., the length of the column 
            vector :math:`\\mathbf{y}(k)`    
        Ts : float
            the sampling interval
            
        Raises
        ------
        :py:class:`mpc.systems.DtSystemError`
            if ``x0.shape != (n_states, 1)``
        """
        # number of states
        self.n_states = n_states
        
        # number of inputs
        self.n_inputs = n_inputs
        
        # number of outputs
        self.n_outputs = n_outputs
        
        # sampling time
        self.Ts = Ts
        
        # check initial condition
        x0 = np.asmatrix(x0)
        if not x0.shape == ( self.n_states, 1):
            raise DtSystemError('wrong shape of initial state vector')
            
        self.x = np.matrix(x0)
    
    def simulate( self, u ):
        """Simulate open-loop system dynamics and get back measurements of the outputs.
        
        Parameters
        ----------
        u : np.array object
            the argument ``u`` can be a a single input or a sequence of inputs.
            In the first case ``u`` must have shape equal to ``(n_inputs, 1)``, 
            while in the former it can have a shape equal to ``(n_inputs, n_steps)``,
            where ``n_steps`` indicate the number of steps of the simulation.
        
        Returns
        -------
        y : np.array object
            the outputs of the system at each time step of the simulation.
            This matrix has shape ``(n_outputs, n_steps)``, where ``n_steps``
            is the number of columns of the input argument ``u``.
            The first column of ``y`` is the output vector *before* that 
            the first element of ``u`` has been applied.
            
        """
        # set data to proper shape
        u = np.asmatrix(u)
        
        # initialize outputs array. Each column is the output at time k.
        y = np.matrix( np.zeros( (self.n_outputs, u.shape[1] ) ) )
        
        # for each time step 
        for i in range(u.shape[1]):
            # get measurements of the system
            y[:,i] = self.measure_outputs()
                    
            # compute new state vector
            self._apply_input( u[:,i] )
            
        return np.array( y )
        
    def measure_outputs( self ):
        """Get outputs, computed from system state.
        
        Derived classes can overload this method for doing
        custom things, such as adding noise, or using non linear
        stuff.
        """
        raise NotImplementedError('Use derived classes instead')
    
    def _apply_input( self, u ) :
        """Apply single input control and update system state.
        
        The reason for the existance of this method is that derived 
        classes can override this method as they want, for example by 
        adding some process noise.
        """
        raise NotImplementedError('Use derived classes instead')


class DtNLSystem( DtSystem ):
    def __init__ ( self, f, g, n_states, n_inputs, n_outputs, Ts, x0):
        """A class for non-linear time-invariant discrete-time systems.
    
        Such system are decribed by the following state equation:
        
        .. math::
        
            \\mathbf{x}(k+1) = \\mathbf{f}(\\mathbf{x}(k), \\mathbf{u}(k) )
        
        and by the output equation:
        
        .. math::
        
            \\mathbf{y}(k) = \\mathbf{g}(\\mathbf{x}(k), \\mathbf{u}(k) )
    
        Parameters
        ----------
        f : any python callable
            the :math:`\\mathbf{f}` function in the state equation.       
            This function must accept two arguments and must return 
            a single argument. Arguments must be numpy.matrix
            objects with appropriate shapes. First argument is the state
            vector :math:`\\mathbf{x}` at time step :math:`k`, 
            second argument is the input vector :math:`\\mathbf{u}`
            at time step :math:`k`.
            
        g : any python callable
            the :math:`\\mathbf{g}` function in the output equation.       
            This function must accept two arguments and must return 
            a single argument. Arguments must be numpy.matrix
            objects with appropriate shapes. First argument is the state
            vector :math:`\\mathbf{x}` at time step :math:`k`, 
            second argument is the input vector :math:`\\mathbf{u}`
            at time step :math:`k`.
            
        n_states : int 
            the number of states of the system
        
        n_inputs : int
            the number of inputs
            
        n_outputs : int
            the number of outputs
            
        Ts : float
            the sampling time
        
        x0 : np.matrix, with shape ``(n_states, 1)
            the initial system state
            
        Attributes
        ----------
        x : np.matrix object with shape ``(n_states, 1)``
            the state of the system at time step :math:`k`
        n_states : int
            the length of the state column vector :math:`\\mathbf{x}(k)`
        n_inputs : int
            the number of system inputs, i.e., the length of the column 
            vector :math:`\\mathbf{u}(k)`
        n_outputs : int
            the number of system outputs, i.e., the length of the column 
            vector :math:`\\mathbf{y}(k)`    
        Ts : float
            the sampling interval
            
        f : python callable
            the system update equation function
            
        g : python callable
            the output equation function
            
        """
        # state equation function
        self.f = f
        
        # outputs equation function
        self.g = g
        
        DtSystem.__init__ ( self, Ts, x0 )


class DtLTISystem( DtSystem ):
    def __init__ ( self, A, B, C, D, Ts, x0 ):
        """A class for linear time-invariant discrete time systems.
        
        The state equation is:
        
        .. math ::
            x_{k+1} = A x_{k} + B u_{k}
            
        while the output equation is:
        
        .. math ::
            y_{k} = C x_{k} + D u_{k}
        
            
        Parameters
        ----------
        A : np.matrix object
            the transition matrix. Must be square with shape equal
            to ``(n_states, n_states)``.
            
        B : np.matrix object
            the input matrix. Must have shape equal to
            ``(n_states, n_inputs)``.
        
        C : np.matrix object
            the output matrix. Must have shape equal to 
            ``(n_outputs, n_states)``.
            
        D : np.matrix object
            the direct output matrix. Must have shape equal to 
            ``(n_outputs, n_inputs)``.
            
        Ts : float
            the sampling time 
            
        x0 : np.matrix object
            the initial conditions state vector
        
        Attributes
        ----------
        A, B, C, D : np.matrix object the matrices of the state 
            and output equations
        
        n_states : int 
            the number of states of the system
        
        n_inputs : int
            the number of inputs
            
        n_outputs : int
            the number of outputs
            
        Ts : float
            the sampling time
            
        Raises
        ------
        mpc.systems.DtSystemError : if system matrices do not have the correct shape
        
        
        """

        # set state-space matrices
        self.A = np.matrix(A, copy=True)
        self.B = np.matrix(B, copy=True)
        self.C = np.matrix(C, copy=True)
        self.D = np.matrix(D, copy=True)
        
        # checks
        if not self.A.shape[0] == self.A.shape[1]:
            raise DtSystemError('matrix A must be square')
        
        if not self.B.shape[0] == self.A.shape[0]:
            raise DtSystemError('matrix B must be have the same number of rows as matrix A')
            
        if not self.C.shape[1] == self.A.shape[0]:
            raise DtSystemError('matrix D must be have the same number of columns as matrix A')
            
        if not self.D.shape[0] == self.C.shape[0]:
            raise DtSystemError('matrix D must be have the same number of rows as matrix C')

        # call parent __init__
        DtSystem.__init__( self, n_outputs = self.C.shape[0],
                                 n_states = self.A.shape[0],
                                 n_inputs = self.B.shape[1],
                                 Ts = Ts,
                                 x0 = x0  )

    def measure_outputs( self ):
        """Get available outputs."""
        return self.C * self.x 
    
    def _apply_input( self, u ) :
        """Apply single input control."""
        self.x = self.A * self.x + self.B * u 


class NoisyDtLTISystem( DtLTISystem ):
    def __init__ ( self, A, B, C, D, Ts, Sw, Sv, x0 ):
        """A class to simulate linear time-invariant, discrete 
        time systems  with addded process and measurement noise.
        
        The state equation is:
        
        .. math ::
            x_{k+1} = A x_{k} + B u_{k} + w_k
            
        while the output equation is:
        
        .. math ::
            y_{k} = C x_{k} + D u_{k} + y_k
        
            
        Parameters
        ----------
        A : np.matrix object
            the transition matrix. Must be square with shape equal
            to ``(n_states, n_states)``.
            
        B : np.matrix object
            the input matrix. Must have shape equal to
            ``(n_states, n_inputs)``.
        
        C : np.matrix object
            the output matrix. Must have shape equal to 
            ``(n_outputs, n_states)``.
            
        D : np.matrix object
            the direct output matrix. Must have shape equal to 
            ``(n_outputs, n_inputs)``.
            
        Ts : float
            the sampling time 
            
        Sw : np.matrix object
            the process error covariance matrix
            
        Sv : np.matrix object
            the measurements error covariance matrix
            
        x0 : np.matrix object
            the initial conditions state vector
        
        Attributes
        ----------
        A, B, C, D : np.matrix object the matrices of the state 
            and output equations
        
        n_states : int 
            the number of states of the system
        
        n_inputs : int
            the number of inputs
            
        n_outputs : int
            the number of outputs
            
        Ts : float
            the sampling time
            
        Sw : np.matrix object
            the process error covariance matrix
            
        Sv : np.matrix object
            the measurements error covariance matrix
            
        Raises
        ------
        mpc.systems.DtSystemError : if system matrices do not have the correct shape
        """
        # set process and measurement noise covariance matrices
        self.Sw = Sw
        self.Sv = Sv
        
        # call parent __init__
        DtLTISystem.__init__( self, A, B, C, D, Ts, x0 )
        
    def _measurement_noise( self ):
        """Get some noise."""
        return np.matrix( np.random.multivariate_normal( np.zeros((self.n_outputs,)), self.Sv, 1 ).reshape( self.n_outputs, 1) )
    
    def _process_noise( self ):
        """Get some noise."""
        return np.matrix( np.random.multivariate_normal( np.zeros((self.n_states,)), self.Sw, 1 ).reshape(self.n_states,1) )
               
    def measure_outputs( self ):
        """
        """
        return self.C * self.x + self._measurement_noise()
    
    def _apply_input( self, u ):
        """
        """
        self.x = self.A * self.x + self.B * u + self._process_noise()


class DoubleIntegrator( DtLTISystem ):
    def __init__ ( self, Ts, x0 ):
        
        # state space matrices of the double integrator
        A = [[1, Ts], [0, 1]]
        B = [[Ts**2/2], [Ts]]
            
        # the state is fully observable
        C = [[1, 0], [0, 1]]
        D = [[0], [0]]
        
        DtLTISystem.__init__( self, A, B, C, D, Ts, x0 )


class DtSystemError( Exception ):
    """Base exception raised when state-space matrices of non
    coherent shape are given."""
    pass


class ObservabilityError( Exception ):
    """Exception raised when trying to get
    state from measurements of a no observable system."""
    pass


def c2d( system, Ts ):
    """Convert continuous-time model to discrete time, 
    using a zero-order hold method."""
    Ad = scipy.linalg.expm(system.A*Ts , q=7)
    Bd = _series( system.A, Ts, 10 )*system.B
    Cd = system.C
    Dd = system.D
    
    return DtLTISystem( Ad, Bd, Cd, Dd, Ts )

def _series( A, Ts, n=10):
    """Matrix power series"""
    S = np.eye(A.shape[0])*Ts
    Am = np.matrix(A)
    for i in xrange(1,n):
        S +=  (Am**i) * Ts**(i+1) / scipy.factorial(i+1)
    return S
