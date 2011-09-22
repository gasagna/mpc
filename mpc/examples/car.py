"""This example shows how to simulate the dynamics of a car which is 
accelerated at constant rate on a straigth. We are measuring the
position of the car with a very noisy GPS, which we do not actually 
trust much. So we have implemented a Kalman filter to estimate the 
position of our car. By the way, a state observer is also implemented 
via the Klaman filtering and we get the estimate of the car's velocity too.

In this example:

+ use kalman state observer, to estimate car's position and velocity.
+ subclass Controller class so that we can mimick the car's throttle.

"""

import numpy as np

from mpc.controllers import Controller
from mpc.systems import NoisyDtLTISystem
from mpc.observers import KalmanStateObserver
from mpc.simulation import SimEnv



class Car( NoisyDtLTISystem ):
    def __init__ ( self, Ts, Sw, Sv, x0 ):
        """Our system is a car, maybe a shiny Ferrari 458 spider!
        We also encapsulate in this class the characteristics of our crappy
        GPS sensor, since the sensor is actually on the car!
        
        (Note that we have inherited from NoisyDtLTISystem.)
        
        Parameters
        ----------
        Ts : float
            the sampling time
            
        Sw : np.matrix
            the process noise covariance matrix 
            
        Sv : np.matrix
            the measurement noise covariance matrix. Our GPS is 
            quite crappy, so this may be quite high. 
            
        x0 : np.matrix, shape = (2,1)
            the initial condition of the car, i.e., position and velocity
            
        """
        
        # state space matrices, you may want to check if i 
        # did not do any mistake in the discretization!
        A = [[1, Ts], [0, 1]]
        B = [[Ts**2/2], [Ts]]
        C = [[1.0, 0.0]]
        D = [[0.0]]
        
        # we call the parent class __init__ method.
        NoisyDtLTISystem.__init__( self, A, B, C, D, Ts, Sw, Sv, x0 )


class Throttle( Controller ):
    def __init__ ( self, value=1 ):
        """This is a nasty trick. Basically we drive on a straight by
        pushing the throttle pedal. In this case, what we are actually 
        doing is to have the full throttle! 
        
        Parameters
        ----------
        value : float
            this is basically the value of the (constant) accelleration 
            we are imposing on the car. Realistic values are 9.81 * 1 for
            a Formula 1 car and 9.81*0.2 for an old Fiat Panda.
        """
        self.value = float(value)
        
    def compute_control_input( self, x ):
        """Always return the same value"""
        return np.matrix([[self.value]])
    
        
if __name__ == '__main__':
    
    # sampling time. 
    Ts = 0.05
    
    # Process and measurement noise covariance matrices
    # this value should be the variance of the measurement position error.
    # Our GPS is quite crappy so i have set a Root Mean Square Error of 
    # 10 meters
    Sv = np.matrix( [[10.0**2]] )
    
    # This is a little bit less intuitive to get.
    # It accounts for the fact that even if we press the pedal to full,
    # we do not get the expected accelleration, because of the fact the 
    # our car is running in an environment with gusts, holes on the road, 
    # ..., which introduce a random variation in the car's accelleration.
    Sw = 1e1 * np.matrix( [[0.25*Ts**4, 0.5*Ts**3],[0.5*Ts**3, Ts**2]] )
    
    # Ok, now define the system, giving zero initial conditions.
    car = Car( Ts=Ts, Sw=Sw, Sv=Sv, x0 = np.matrix( [[0.0],[0.0]] ) )
    
    # define controller. We want a constant value of input. Remember to 
    # put a value for the accelleration in m/s^2, and not g!
    throttle = Throttle( value=9.81*1 ) 
    
    # create kalman state observer object
    kalman_observer = KalmanStateObserver( car )
    
    # Create simulator.
    # This is our simulation obejct; it is responsible of making all 
    # the parts communicating togheter.
    sim = SimEnv( car, throttle, kalman_observer )
        
    # run simulation for 10 seconds
    res = sim.simulate( 10 )
        
    # now do plots
    from pylab import *
    
    subplot(311)
    plot ( res.t, res.y[0], 'b.', label='Position measured' )
    plot ( res.t, res.x[0], 'r-', lw=2, label='Position estimated' )
    legend(loc=2, numpoints=1)
    ylabel( 'x [m]' )
    grid()
    
    subplot(312, sharex=gca())
    plot ( res.t, res.x[1], 'r-', label='Velocity estimated' )
    ylabel( 'v [m/s]' )
    legend(loc=2, numpoints=1)
    grid()
    
    subplot(313, sharex=gca())
    plot ( res.t, res.u[0], 'b-', label='Input' )
    xlabel( 't [s]' )
    ylabel( 'u' )
    legend(loc=2, numpoints=1)
    grid()
    
    show()
