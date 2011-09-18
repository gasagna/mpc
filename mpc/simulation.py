"""A code to simulate Discrete Time Linear Time Invariant Dynamical systems
controlled with a Linear Quadratic Regulator and Model Predictive Controller."""

__author__ = 'Davide Lasagna'
__date__ = '26/07/2011'


import numpy as np

from mpc.systems import KalmanFilter


class SimEnv( object ):
    """A simulation environment for discrete-time dynamic systems."""
    def __init__ ( self, system, controller=None, use_state_observer=True ):
        
        # the system we want to simulate
        self.system = system
        
        # the controller ( state feedback )
        self.controller = controller
        
        # use kalman filter?
        self._use_state_observer = use_state_observer
        
        if self._use_state_observer:
            self.kalman_filter = KalmanFilter( system, x0=system.x  )
    
    def simulate( self, Tsim, u_func=None ):
        """Simulate controlled system dynamics."""
        
        # run for sufficient steps
        n_steps = int( Tsim / self.system.Ts ) + 1
        
        # Preallocate matrices
        # state estimate
        xhat = np.zeros( (self.system.n_states, n_steps) )
        xhat[:,0] = self.system.x.ravel()
        
        # control input
        u = np.zeros( (self.system.n_inputs, n_steps) )
        
        # measurements
        y = np.zeros( (self.system.n_outputs, n_steps) )
        
        # run simulation
        for k in xrange( n_steps-1 ):
            
            # get measuremts
            y[:,k] = self.system.measure_outputs().ravel()
            
            # compute control move based on the state at this time. 
            # Futhermore apply some function to the output, if needed.
            if u_func:
                u[:,k] = u_func( self.controller.compute_control_input( xhat[:,k].reshape(self.system.n_states,1) ) )
            else:
                u[:,k] = self.controller.compute_control_input( xhat[:,k].reshape(self.system.n_states,1))
            
            # apply input 
            self.system._apply_input( u[:,k].reshape(self.system.n_inputs, 1) )
            
            # now i am at step step k+1
            # estimate state using kalman filter using output at current 
            # step and previous control input value
            if self._use_state_observer:
                xhat[:,k+1] = self.kalman_filter.estimate( y[:,k], u[:,k] ).ravel()
            else:
                xhat[:,k+1] = self.system.x.ravel()
                
            
        return SimulationResults(xhat, u, y, self.system.Ts)


class SimulationResults():
    def __init__ ( self, x, u, y, Ts ):
        self.x = x
        self.u = u
        self.y = y
        self.t = np.arange(x.shape[1]) * Ts
