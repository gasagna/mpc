"""A code to simulate Discrete Time Linear Time Invariant Dynamical systems
controlled with a Linear Quadratic Regulator and Model Predictive Controller."""

__author__ = 'Davide Lasagna'
__date__ = '26/07/2011'


import numpy as np
import scipy
import scipy.linalg


class DtSystemError( Exception ):
    """Basic exception raised when wrong state-space matrices are given"""
    pass


class DtSystem( object )
    """Base class of all the  discrete time systems in this module. Use 
    derived classes to work, cause this is just the skeleton of the structure."""
    def __init__ ( self, n_states, n_inputs, n_outputs, Ts, x0 ):
        # number of states
        self.n_states = n_states
        
        # number of inputs
        self.n_inputs = n_inputs
        
        # number of outputs
        self.n_outputs = n_outputs
        
        # sampling time
        self.Ts = Ts
        
        # check initial condition
        if not x0.shape == ( self.n_states, 1):
            raise DtSystemError('wring shape of initial state vector')
            
        self.x = np.matrix(x0)
    
    def simulate( self, u ):
        """Simulate open-loop system dynamics and get back measurements of the outputs.
        
        Parameters
        ----------
        u : np.matrix object
            the argument ``u`` can be a a single input or a sequence of inputs.
            In the first case ``u`` must have shape equal to ``(n_inputs, 1)``, 
            while in the former it can have a shape equal to ``(n_inputs, n_steps)``,
            where ``n_steps`` indicate the number of steps of the simulation.
        
        Returns
        -------
        y : np.matrix object
            the output value
        """
        
        # initialize outputs array. Each column is the output at time k.
        y = np.zeros( (self.n_outputs, u.shape[1] ) )
        
        # for each time step 
        for i in range(u.shape[1]):
            
            # compute new state vector
            self.x = self._apply_input( u[:,i] )
        
            # get measurements of the system. (but this is already at time k+1: i is an index over array indices )
            y[:,i] = self.measure_outputs()
        
        return y
        
    def measure_outputs( self ):
        """Get available outputs, computed from system state.
        
        Derived classe can overload this method for doing
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
    """Base class for non-linear time-invariant discrete-time systems."""
    def __init__ ( self, f, g, n_states, n_inputs, n_outputs, Ts, x0):
        """ """
        # state equation function
        self.f = f
        
        # outputs equation function
        self.g = g
        
        System.__init__ ( self, Ts, x0 )
        
    def measure_outputs( self ):
        """Get available outputs, computed from system state."""
        return self.g( self.x )
    
    def _apply_input( self, u ) :
        """Apply single input control and update system state."""
        return self.f( self.x, u )


class DtLTISystem( DtSystem ):
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

        # call parent __init__
        DtSystem.__init__( self, n_outputs = self.C.shape[0],
                                 n_states = self.A.shape[0],
                                 n_inputs = self.B.shape[1]
                                 Ts = Ts,
                                 x0 = x0  )

    def measure_outputs( self ):
        """Get available outputs.
        
        Derived classe can overload this method for doing
        custom things, such as adding noise.
        
        """
        return self.C * self.x 
    
    def _apply_input( self, u ) :
        """Apply single input control.
        
        The reason for the existance of this method is that derived 
        classes can override this method as they want, for example by 
        adding some process noise .
        """
        return self.A * self.x + self.B * u 


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
        return self.A * self.x + self.B * u + self._process_noise()


class DoubleIntegrator( NoisyDtLTISystem ):
    def __init__ ( self, Ts, Sw, Sv, x0 ):
        
        # state space matrices
        A = [[1, Ts], [0, 1]]
        B = [[Ts**2/2], [Ts]]
        C = [[1, 0]]
        D = [[0]]
        
        NoisyDtLTISystem.__init__( self, A, B, C, D, Ts, Sw, Sv, x0 )


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

