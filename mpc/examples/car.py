import numpy as np

from mpc.controllers import Controller
from mpc.systems import NoisyDtLTISystem
from mpc.simulation import SimEnv


class Car( NoisyDtLTISystem ):
    def __init__ ( self, Ts, Sw, Sv, x0 ):
        
        # state space matrices
        A = [[1, Ts], [0, 1]]
        B = [[Ts**2/2], [Ts]]
        C = [[1, 0]]
        D = [[0]]
        
        NoisyDtLTISystem.__init__( self, A, B, C, D, Ts, Sw, Sv, x0 )


class Throttle( Controller ):
    def __init__ ( self, value=1 ):
        """
        This is how much input we give
        """
        self.value = float(value)
        
    def compute_control_input( self, x ):
        """Always return the same value"""
        return np.matrix([[self.value]])
    
        
if __name__ == '__main__':
    
    # sampling time
    Ts = 5e-3
    
    # process and measurement noise
    Sv = np.matrix( [[10]] )
    Sw = 10 * np.matrix( [[0.25*Ts**4, 0.5*Ts**3],[0.5*Ts**3, Ts**2]] )
    
    # define system
    car = Car( Ts=Ts, Sw=Sw, Sv=Sv, x0 = np.array( [[0.0],[0.0]] ) )
    
    # define controller. We want a constant value of input
    throttle = Throttle( value=5 ) 
    
    # create simulator
    sim = SimEnv( car, throttle )
        
    # run simulation
    res = sim.simulate( 5 )
        
        
    # do plots
    from pylab import *
    
    subplot(211)
    plot ( res.t, res.x[0], '-' )
    plot ( res.t, res.y[0], '--' )
    grid()
    
    subplot(212, sharex=gca())
    plot ( res.t, res.x[1] )
    grid()
    
    
    show()
