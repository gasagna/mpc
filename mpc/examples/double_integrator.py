import numpy as np

from mpc.controllers import LQController
from mpc.systems import NoisyDtLTISystem
from mpc.simulation import SimEnv

        
class DoubleIntegrator( NoisyDtLTISystem ):
    def __init__ ( self, Ts, Sw, Sv, x0 ):
        
        # state space matrices
        A = [[1, Ts], [0, 1]]
        B = [[Ts**2/2], [Ts]]
        C = [[1, 0]]
        D = [[0]]
        
        NoisyDtLTISystem.__init__( self, A, B, C, D, Ts, Sw, Sv, x0 )
        
if __name__ == '__main__':
    
    Ts = 5e-3
    
    Sv = np.matrix( [[4]] )
    Sw = 100 * np.matrix( [[0.25*Ts**4, 0.5*Ts**3],[0.5*Ts**3, Ts**2]] )
    
    # define system
    di = DoubleIntegrator( Ts=Ts, Sw=Sw, Sv=Sv, x0 = np.matrix( [[10.0],[0.0]] ) )
    
    # define controller
    #nl_controller = NonLinearController( K = 50, alpha=0.5 )
    controller = LQController( di, Q=100*np.eye(2), R=1*np.eye(1)) 
        
    # create simulator
    sim = SimEnv( di, controller )
    
    # run
    res = sim.simulate( 8 )
        
    # plot results
    from pylab import *
    
    subplot(311)
    plot ( res.t, res.x[0], '-' )
    plot ( res.t, res.y[0], '--' )
    grid()
    
    subplot(312, sharex=gca())
    plot ( res.t, res.x[1] )
    grid()
    
    subplot(313, sharex=gca())
    plot ( res.t, res.u[0] )
    grid()
    
    
    show()
