"""A code to simulate Discrete Time Linear Time Invariant Dynamical systems
controlled with a Linear Quadratic Regulator and Model Predictive Controller."""

__author__ = 'Davide Lasagna'
__date__ = '26/07/2011'


import numpy as np
import scipy
import scipy.linalg


class LTISystemError( Exception ):
    """Basic exception raised when wrong state-space matrices are given"""
    pass


class DtNLSystem( object ):
    """In this class i will implement a non-lineardiscrete-time,
    time-invariant system. 
    
    The idea is to instantiate from this class by giving as argument a 
    non linear function which is stored internally and which is fed at 
    each sampling time with the input vector. Then the function is 
    evaluated and the outputs are returned. TODO.
    """
    pass

class DtLTISystem( object ):
    def __init__ ( self, A, B, C, D, Ts, x0 ):
        """A class for linear time-invariant discrrete time systems.
        
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
        LTISystemError : if system matrices do not have the correct shape
        
        
        """

        # set state-space matrices
        self.A = np.matrix(A, copy=True)
        self.B = np.matrix(B, copy=True)
        self.C = np.matrix(C, copy=True)
        self.D = np.matrix(D, copy=True)
        
        # checks
        if not self.A.shape[0] == self.A.shape[1]:
            raise LTISystemError('matrix A must be square')
        
        if not self.B.shape[0] == self.A.shape[0]:
            raise LTISystemError('matrix B must be have the same number of rows as matrix A')
            
        if not self.C.shape[1] == self.A.shape[0]:
            raise LTISystemError('matrix c must be have the same number of columns as matrix A')
            
        if not self.D.shape[0] == self.C.shape[0]:
            raise LTISystemError('matrix d must be have the same number of rows as matrix C')
        
        # number of states
        self.n_states = self.A.shape[0]
        
        # number of inputs
        self.n_inputs = self.B.shape[1]
        
        # number of outputs
        self.n_outputs = self.C.shape[0]
        
        # sampling time
        self.Ts = Ts
        
        # set initial condition
        self._x = np.matrix( x0 ) 
        
        if not self._x.shape == ( self.n_states, 1):
            raise LTISystemError('wring shape of initial state vector')
        
    def simulate( self, u ):
        """Simulate system and get back measurements.
        
        Parameters
        ----------
        u : np.matrix object
            the control input
        
        Returns
        -------
        y : np.matrix object
            the output value
        """
        
        # compute new state vector
        x_new = self.A * self._x + self.B * u 
        
        # get measurements of the system
        y = self.C * x_new 
        
        # update attribute for the next call
        self._x = x_new
        
        return y.reshape( self.n_outputs, 1 )


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
        LTISystemError : if system matrices do not have the correct shape
        """
        # create the LTI system definition
        DtLTISystem.__init__( self, A, B, C, D, Ts, x0  )
        
        # set process and measurement noise
        self.Sw = Sw
        self.Sv = Sv
        
    def simulate( self, u ):
        """Simulate system and get back noisy measurements of the outputs.
        
        Parameters
        ----------
        u : np.matrix object
            the control input
        
        Returns
        -------
        y : np.matrix object
            the output value
        """
        
        # compute new state vector
        x_new = self.A * self._x + self.B * u + np.matrix( np.random.multivariate_normal( np.zeros((self.n_states,)), self.Sw, 1 ).reshape(self.n_states,1) )
        
        # get measurements of the system
        y = self.C * x_new + np.matrix( np.random.multivariate_normal( np.zeros((self.n_outputs,)), self.Sv, 1 ).reshape( self.n_outputs, 1) )
        
        # update attribute for the next call
        self._x = x_new
        
        return y


class KalmanFilter( object ):
    def __init__ ( self, system, x0=None ):
        
        # set attribute
        self.system = system
        
        # set initial condition for state estimate
        if x0 is None:
            self._xhat = system._x + np.matrix( np.random.multivariate_normal( np.zeros((system.n_states,)), system.Sw, 1 ).reshape(system.n_states,1) )
        else:
            self._xhat = x0
        
        # covariance matrix of the state estimate
        self.P = self.system.Sw
        
    def estimate( self, y, u_old ):
        
        #simulate system with state estimate at previous step
        xhat = self.system.A * self._xhat + self.system.B * u_old
        
        # form the innovation vector
        inn = y - self.system.C*xhat
        
        # compute the covariance of the innovation
        s = self.system.C*self.P*self.system.C.T + self.system.Sv
        
        # form the kalman gain matrix
        K = self.system.A*self.P*self.system.C.T * np.linalg.inv(s)
        
        # update state estimate
        xhat += K*inn
        
        # compute covariance of the estimation error
        self.P = self.system.A*self.P*self.system.A.T -  \
                 self.system.A*self.P*self.system.C.T*np.linalg.inv(s)*\
                 self.system.C*self.P*self.system.A.T + self.system.Sw
        
        # update state estimate for next iteration
        self._xhat = xhat
        
        # return state estimate
        return xhat


def c2d( system, Ts, method='euler-forward' ):
    """Convert continuous-time model to discrete time, 
    using a zero-order hold method."""
    Ad = scipy.linalg.expm(system.A*Ts , q=7)
    Bd = _series( system.A, Ts, 10 )*system.B
    Cd = system.C
    Dd = system.D
    
    return DtLTISystem( Ad, Bd, Cd, Dd, Ts )

def _series( A, Ts, n=10):
    """MAtrix power series"""
    S = np.eye(A.shape[0])*Ts
    Am = np.matrix(A)
    for i in xrange(1,n):
        S +=  (Am**i) * Ts**(i+1) / scipy.factorial(i+1)
    return S

