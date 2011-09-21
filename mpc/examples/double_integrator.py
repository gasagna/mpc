import numpy as np
from pylab import *

from mpc.controllers import LQController
from mpc.systems import NoisyDtLTISystem
from mpc.observers import KalmanStateObserver
from mpc.simulation import SimEnv

        
class DoubleIntegrator( NoisyDtLTISystem ):
    def __init__ ( self, Ts, Sw, Sv, x0 ):
        
        # state space matrices
        A = [[1, Ts], [0, 1]]
        B = [[Ts**2/2], [Ts]]
        C = [[1, 0], [0, 1]]
        D = [[0], [0]]
        
        NoisyDtLTISystem.__init__( self, A, B, C, D, Ts, Sw, Sv, x0 )
        
if __name__ == '__main__':
    
    Ts = 1.0 / 400
    
    Sv = np.matrix( [[1, 0], [0, 1]] )
    Sw = 1 * np.matrix( [[0.25*Ts**4, 0.5*Ts**3],[0.5*Ts**3, Ts**2]] )
    
    # define system
    di = DoubleIntegrator( Ts=Ts, Sw=Sw, Sv=Sv, x0 = np.matrix( [[10.0],[5.0]] ) )
    
    # define controller
    controller = LQController( di, Q=10*np.eye(2), R=1*np.eye(1)) 
    
    # create kalman state observer
    kalman_observer = KalmanStateObserver( di, x0=np.matrix([[10.0],[5.0]]) )
        
    # create simulator
    sim = SimEnv( di, controller, observer=kalman_observer )
    
    # run
    res = sim.simulate( 10 )
        
    # plot results
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
    
    print 1
    show()
    print 2
