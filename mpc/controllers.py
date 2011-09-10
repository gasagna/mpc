"""A code to simulate Discrete Time Linear Time Invariant Dynamical systems
controlled with a Linear Quadratic Regulator and Model Predictive Controller."""

__author__ = 'Davide Lasagna'
__date__ = '26/07/2011'


import numpy as np
import pydare as dare


class Controller( object ):
    """A controller is an object with a main method, ``compute_control_input``,
    which is responsible of returning the appropriate control input, based on the
    system's state."""
    def compute_control_input( self, x ):
        raise NotImplemented( 'Call derived classes ')


class LQController( Controller ):
    def __init__ ( self, system, Q, R ):
        """A class to build an Infinite Horizon Linear Quadratic Controller."""
        
        # some definitions
        self.Q = np.matrix( Q )
        self.R = np.matrix( R )
        
        # solve Algebraic Riccati Equation
        P = dare.DareSolver( system.A, system.B, self.Q, self.R ).solve_slycot()
        
        # create static state feedback matrix
        self.K  = (self.R + system.B.T * P * system.B).I * (system.B.T * P * system.A)
        
    def compute_control_input( self, x ):
        """Compute static state feedback."""
        return - np.dot( self.K, x)
