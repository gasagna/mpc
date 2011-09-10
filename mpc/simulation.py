"""A code to simulate Discrete Time Linear Time Invariant Dynamical systems
controlled with a Linear Quadratic Regulator and Model Predictive Controller."""

__author__ = 'Davide Lasagna'
__date__ = '26/07/2011'


import numpy as np

from mpc.systems import KalmanFilter


class SimEnv( object ):
    """A simulation environment for discrete-time dynamic systems."""
    def __init__ ( self, system, controller=None, kalman_filter=True ):
        
        # the noisy system we want to simulate
        self.system = system
        
        # the controller ( state feedback )
        self.controller = controller
        
        self.kalman_filter = KalmanFilter( system, system._x_old )
    
    def simulate( self, Tsim ):
        """Simulate controlled system dynamics."""
        
        # run for sufficient steps
        n_steps = int( Tsim / self.system.Ts ) + 1
        
        # Preallocate matrices
        # state estimate
        xhat = np.zeros( (self.system.n_states, n_steps) )
        
        # control input
        u = np.zeros( (self.system.n_inputs, n_steps) )
        
        # measurements
        y = np.zeros( (self.system.n_outputs, n_steps) )
        
        # set initial condition
        xhat[:,0] = self.system._x_old[:,0]
        
        # run simulation
        for k in xrange( n_steps-1 ):
            
            # now i am at step k
            # Simulation step
            # if at step k i apply command u[:,k] i will obtain at 
            # step k+1 the output y[:,k+1]
            y[:,k+1] = self.system.sim( u[:,k] )[:,0]
            
            # now i am at step step k+1
            # estimate state using kalman filter using output at current 
            # step and previous control input value
            xhat[:,k+1] = self.kalman_filter.estimate( y[:,k+1], u[:,k] ).ravel()
        
            # compute control move for next step, at next iteration, based
            # on the state at this time 
            u[:,k+1] = self.controller.compute_control_input( xhat[:,k].reshape(self.system.n_states,1) )
            
            
        return SimulationResults(xhat, u, y, self.system.Ts)


class SimulationResults():
    def __init__ ( self, x, u, y, Ts ):
        self.x = x
        self.u = u
        self.y = y
        self.t = np.arange(x.shape[1]) * Ts
