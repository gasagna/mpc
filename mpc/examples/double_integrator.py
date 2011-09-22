"""This example shows the following things:

+ How to subclass the DtLTISystem class to create
  a double integrator system;
+ How to set up an infinite horizon Linear-Quadratic controller;
+ How to simulate the behaviour of the system.
+ How to use the mpc.simulation.SimulationResults 
  class to plot the results of the simulation.
"""

import numpy as np
from pylab import *

from mpc.systems import DtLTISystem
from mpc.controllers import LQController
from mpc.simulation import SimEnv

        
class DoubleIntegrator( DtLTISystem ):
    def __init__ ( self, Ts, x0 ):
        
        # state space matrices of the double integrator
        A = [[1, Ts], [0, 1]]
        B = [[Ts**2/2], [Ts]]
        
        # the state is fully observable
        C = [[1, 0], [0, 1]]
        D = [[0], [0]]
        
        DtLTISystem.__init__( self, A, B, C, D, Ts, x0 )
        
if __name__ == '__main__':
    
    # this is the sampling time
    Ts = 1.0 / 40
    
    # initial condition
    x0 = np.matrix( [[-6.0],[0.0]] )
    
    # define system
    di = DoubleIntegrator( Ts=Ts, x0=x0 )
    
    # define controller
    controller = LQController( di, Q=1*np.eye(2), R=0.1*np.eye(1) ) 
    
    # create simulator
    sim = SimEnv( di, controller )
    
    # run simulation
    res = sim.simulate( 6 )
        
    # plot results
    subplot(311)
    plot ( res.t, res.x[0], '-' )
    ylabel('x [m]')
    grid()
    
    subplot(312, sharex=gca())
    plot ( res.t, res.x[1] )
    ylabel('v [m/s]')
    grid()
    
    subplot(313, sharex=gca())
    plot ( res.t, res.u[0] )
    xlabel('t [s]')
    ylabel('u')
    grid()
    
    show()
