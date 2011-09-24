"""This example shows the following things:

+ simulate single integrator system behaviour
  with process noise using a non linear controller.
  
+ subclass mpc.controllers.Controller to enforce 
  saturation of the input.
  
We want to zero regulate the system.
"""

import numpy as np

from mpc.controllers import Controller
from mpc.systems import NoisyDtLTISystem
from mpc.simulation import SimEnv


class NonLinearController( Controller ):
    def __init__ ( self, K, alpha, saturation ):
        """We create a custom controller object.
        
        Control input is a static state feedback
        of the form: 
            u = K * |x|^alpha * sign(x)
        where x is the state and K and alpha are 
        to constants.
        """
        self.K = K
        self.alpha = alpha
        self.saturation = saturation
        
    def compute_control_input( self, x ):
        """The control input can saturate."""
        u = -self.K * np.abs(x)**self.alpha * np.sign(x)
        return np.clip(u, self.saturation[0], self.saturation[1])
        
        
class SingleIntegrator( NoisyDtLTISystem ):
    def __init__ ( self, Ts, w0, x0 ):
        """Subclass NoisyDtLTISystem class
        and we add parameter w0, the variance 
        of the process noise.
        """
        
        # state space matrices
        # We can create lists of lists,
        # then internally they will be converted
        # to numpy matrices
        A = [[1]]
        B = [[Ts]]
        C = [[1]]
        D = [[0]]
        
        # process noise and measurement noise. Measurement noise
        # is set to be very low.
        Sw = np.matrix( [[w0]] )
        Sv = np.matrix( [[1e-9]] )
        
        NoisyDtLTISystem.__init__( self, A, B, C, D, Ts, Sw, Sv, x0 )
        
if __name__ == '__main__':
    
    # define system
    single_integ = SingleIntegrator( Ts=10e-3, w0=1e-4, x0=np.matrix([[5]]) )
    
    # define non-linear controller subject to saturation
    nl_controller = NonLinearController( K = 50, alpha=0.9, saturation=(-4, 0.5) )
    
    # create simulator
    sim = SimEnv( single_integ, controller=nl_controller )

    # simulate system for one second
    res = sim.simulate( 5.0 )

    # do plots
    from pylab import *
    
    subplot(211)
    plot ( res.t, res.x[0] )
    ylabel('e [m]')
    grid()
    
    subplot(212, sharex=gca())
    plot ( res.t, res.u[0] )
    ylabel('u')
    xlabel('t [s]')
    grid()
    
    
    show()
