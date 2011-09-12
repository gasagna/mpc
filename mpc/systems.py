"""A code to simulate Discrete Time Linear Time Invariant Dynamical systems
controlled with a Linear Quadratic Regulator and Model Predictive Controller."""

__author__ = 'Davide Lasagna'
__date__ = '26/07/2011'


import numpy as np
import scipy
import scipy.linalg


class LTISystemError( Exception ):
    """Basic exception raised when state-space matrices are given"""
    pass


class DtLTISystem( object ):
    def __init__ ( self, A, B, C, D, Ts ):
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
        
        self.Ts = Ts
        
    def sim( self, u ):
        """Simulate system and get back noisy measurements of the outputs."""
        
        # compute new state vector
        x_new = self.A * self._x_old + self.B * u 
        
        # get measurements of the system
        y = self.C * x_new 
        
        # update attribute for the next call
        self._x_old = x_new
        
        return y.reshape( self.n_outputs, 1 )


class NoisyDtLTISystem( DtLTISystem ):
    def __init__ ( self, A, B, C, D, Ts, Sw, Sv, x0 ):
        """A class to simulate linear time-invariant, discrete 
        time systems  with addded process and measurement noises
        """
        # create the LTI system definition
        DtLTISystem.__init__( self, A, B, C, D, Ts  )
        
        # set process and measurement noise
        self.Sw = Sw
        self.Sv = Sv
        
        # set initial condition of the system
        self._x_old = np.matrix( x0 ) 
        
        if not self._x_old.shape == ( self.n_states, 1):
            raise LTISystemError('wring shape of initial state vector')
        
    def sim( self, u ):
        """Simulate system and get back noisy measurements of the outputs."""
        
        # compute new state vector
        x_new = self.A * self._x_old + self.B * u + np.random.multivariate_normal( np.zeros((self.n_states,)), self.Sw, 1 ) 
        
        # get measurements of the system
        y = self.C * x_new + np.random.multivariate_normal( np.zeros((self.n_outputs,)), self.Sv, 1 ) 
        
        # update attribute for the next call
        self._x_old = x_new
        
        return y.reshape( self.n_outputs, 1 )


class KalmanFilter( object ):
    def __init__ ( self, system, x0 ):
        
        # set attribute
        self.system = system
        
        # set initial condition for state estimate
        self._xhat_old = x0
        
        # covariance matrix of the state estimate
        self.P = self.system.Sw
        
    def estimate( self, y, u_old ):
        
        #simulate system with state estimate at previous step
        xhat = self.system.A * self._xhat_old + self.system.B * u_old
        
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
        self._xhat_old = xhat
        
        # return state estimate
        return xhat.reshape( self.system.n_states, 1)


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

