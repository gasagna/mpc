import numpy as np

from mpc.controllers import LQController, MPController
from mpc.systems import NoisyDtLTISystem
from mpc.simulation import SimEnv

from pylab import *

        
class DoubleIntegrator( NoisyDtLTISystem ):
    def __init__ ( self, Ts, Sw, Sv, x0 ):
        
        Ts = float(Ts)
        
        # state space matrices
        A = [[1, Ts], [0, 1]]
        B = [[Ts**2/2], [Ts]]
        C = [[1, 0], [0, 1]]
        D = [[0], [0]]
        
        NoisyDtLTISystem.__init__( self, A, B, C, D, Ts, Sw, Sv, x0 )
        
if __name__ == '__main__':
    
    Ts = 0.1
    
    Sv = np.matrix( [[0.00001, 0], [0, 0.00001]] )
    Sw = 0.000001 * np.matrix( [[0.25*Ts**4, 0.5*Ts**3],[0.5*Ts**3, Ts**2]] )
    
    # define system
    di = DoubleIntegrator( Ts=Ts, Sw=Sw, Sv=Sv, x0 = np.matrix( [[-6.0],[0.0]] ) )
    
    # define controller properties
    Q = 1*np.eye(2)
    Q[1,1] = 0.0
    R = 0.1*np.eye(1)
    Hps = np.arange(4,100,5)

    lqcontroller = LQController( di, Q=Q, R=R) 
   
    for Hp in Hps:
        
        di = DoubleIntegrator( Ts=Ts, Sw=Sw, Sv=Sv, x0 = np.matrix( [[-6.0],[0.0]] ) )
        mpcontroller = MPController( di, Q=Q, R=R, Hp=Hp, Hw=1, Hc=Hp) 
        print mpcontroller.K
        
        # create simulator
        sim = SimEnv( di, mpcontroller, use_state_observer=False )
        res = sim.simulate( 10  )
            
        # plot results
        subplot(311)
        plot ( res.t, res.x[0], '-', label='Hp = %d' % Hp )
        legend()
        grid(1)
        
        subplot(312, sharex=gca())
        plot ( res.t, res.x[1], label='Hp = %d' % Hp )
        legend()
        grid(1)
        
        subplot(313, sharex=gca())
        plot ( res.t, res.u[0], label='Hp = %d' % Hp )
        legend()
        grid(1)
    

    show()
