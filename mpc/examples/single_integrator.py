import numpy as np

from mpc.controllers import Controller, LQController
from mpc.systems import NoisyDtLTISystem
from mpc.simulation import SimEnv


class NonLinearController( Controller ):
    def __init__ ( self, K, alpha ):
        self.K = K
        self.alpha = alpha
        
    def compute_control_input( self, x ):
        return -self.K * np.abs(x)**self.alpha * np.sign(x)
        
        
class SingleIntegrator( NoisyDtLTISystem ):
    def __init__ ( self, Ts, w0, x0 ):
        
        # state space matrices
        #  we can create lists of lists
        # then internally they will be converted
        # to numpy matrices
        A = [[1]]
        B = [[Ts]]
        C = [[1]]
        D = [[0]]
        
        # process noise and measurement noise
        Sw = np.matrix( [[w0]] )
        Sv = np.matrix( [[0.00001]] )
        
        NoisyDtLTISystem.__init__( self, A, B, C, D, Ts, Sw, Sv, x0 )
        
if __name__ == '__main__':
    
    
    # define system
    single_integ = SingleIntegrator( Ts=1e-3, w0=10e-4, x0=np.matrix([[5]]) )
    
    # define controller
    nl_controller = NonLinearController( K = 5, alpha=0.1 )
    
    # create simulator
    sim = SimEnv( single_integ, controller=nl_controller )

    # simulate system for one second
    res = sim.simulate( 1.0 )

    # do plots
    from pylab import *
    
    subplot(211)
    plot ( res.t, res.x[0] )
    grid()
    
    subplot(212, sharex=gca())
    plot ( res.t, res.u[0] )
    grid()
    
    
    show()
