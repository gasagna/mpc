__author__ = 'Davide Lasagna, Politecnico di Torino Dipartimento di Ingegneria Aerospaziale. <davide.lasagna@polito.it>'
__date__ = '26/07/2011'
__licence_ = """
Copyright (C) 2011  Davide Lasagna

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__doc__ = """This module contains tools for the simulation of system dynamics.


Summary of classes
==================
.. currentmodule:: mpc.simulation

.. autosummary::
   :toctree: generated
    mpc.observers.SimEnv
    mpc.observers.SimulationResults
"""

import numpy as np


class SimEnv( object ):
    def __init__ ( self, system, controller=None, observer=None ):
        """A simulation environment for discrete-time dynamic systems.
        
        This class provides an environment for the simulation the 
        combination of system, controller and observer.
        
        Parameters
        ----------
        system : an instance of :py:`mpc.systems.DtSystem` or one of 
            its derived classes. This is the system we want to simulate.
            
        controller : an instance of :py:`mpc.systems.Controller` or one
            of its derived classes. The feedback controller. 
            
        observer : 
            a state observer

        Methods
        -------
        simulate : run simulation
        
        """
        # the system we want to simulate
        self.system = system
        
        # the controller ( state feedback )
        self.controller = controller

        # the state observer
        self.observer = observer
    
    def simulate( self, Tsim ):
        """Simulate controlled system dynamics.
        
        Paramters
        ---------
        Tsim : float
            the length of the simulation, in seconds.
            
        Returns
        -------
        res : an instance of :pt:`mpc.simulation.SimulationResults`
            the simulation results.
    
        """
        
        # run for sufficient steps
        n_steps = int( Tsim / self.system.Ts ) + 1
        
        # Preallocate matrices
        # This matrix is for the state 
        xhat = np.zeros( (self.system.n_states, n_steps) )
        
        # control input
        u = np.zeros( (self.system.n_inputs, n_steps) )
        
        # measurements
        y = np.zeros( (self.system.n_outputs, n_steps) )
        
        # run simulation
        for k in xrange( n_steps-1 ):
            
            # get measuremts
            y[:,k] = self.system.measure_outputs().ravel()
            
            # compute control move based on the state at this time. 
            u[:,k] = self.controller.compute_control_input( xhat[:,k].reshape(self.system.n_states,1) )
            
            # apply input 
            self.system._apply_input( u[:,k].reshape(self.system.n_inputs, 1) )
            
            # now we are at step step k+1
            # estimate state using observer based on output at current 
            # step and previous control input value
            if self.observer:
                xhat[:,k+1] = self.observer.get_state_estimate( y[:,k], u[:,k] ).ravel()
            else:
                xhat[:,k+1] = self.system.x.ravel()
                
        return SimulationResults(xhat, u, y, self.system.Ts)


class SimulationResults():
    def __init__ ( self, x, u, y, Ts ):
        """A data container for simulation results.
        
        Attributes
        ----------
        x : np.ndarray, with shape ``(n_states, N)``, where ``N`` is the 
            total number of simuation steps. The time evolution of the 
            system state. If a state observer was used, this is the state
            estimate.
        u : np.ndarray, with shape ``(n_inputs, N)``.
            the time evolution of the control input
        y : np.ndarray, with shape ``(n_outputs, N)``.
            the time evolution of the system's outputs.
        """
        self.x = x
        self.u = u
        self.y = y
        self.t = np.arange(x.shape[1]) * Ts
