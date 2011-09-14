import numpy as np

from mpc.controllers import MPController
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
    
    Ts = 1.0 / 400
    
    Sv = np.matrix( [[0.1]] )
    Sw = 0.001 * np.matrix( [[0.25*Ts**4, 0.5*Ts**3],[0.5*Ts**3, Ts**2]] )
    
    # define system
    di = DoubleIntegrator( Ts=Ts, Sw=Sw, Sv=Sv, x0 = np.matrix( [[10.0],[-5.0]] ) )
    
    # define controller
    #nl_controller = NonLinearController( K = 50, alpha=0.5 )
    controller = MPController( di, Q=1000*np.eye(2), R=0.01*np.eye(1), Hp=20, Hc=20) 
    print controller.K
        
    # create simulator
    sim = SimEnv( di, controller )
    
    def saturation( u ):
        return np.clip(u, -3, 3)
    
    # run
    res = sim.simulate( 1  )
        
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
