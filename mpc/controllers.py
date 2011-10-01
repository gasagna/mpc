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

__doc__ = """This module contains classes representing controllers to be 
used for controlling linear time invariant systems, such as the 
Linear-Quadratic controller (LQ) or Model Predictive Controller (MPC).

A controller is a python object which usually has a single public method, 
``compute_control_input``, which is responsible of returning the 
appropriate control input, based on the system's state, which is
given as argument to the method. For consistency all controllers
inherits for the base class :py:mod:`mpc.controllers.Controller` and you do
it as well if you want to create your own controller class.

Classes
==================
.. currentmodule:: mpc.controllers

.. autosummary::
   :toctree: generated

    mpc.controllers.Controller
    mpc.controllers.LQController
    
.. autoclass:: mpc.controllers.Controller
    :members:
    :inherited-members:
    :show-inheritance:
    
.. autoclass:: mpc.controllers.LQController
    :members:
    :inherited-members:
    :show-inheritance:
"""

import numpy as np
from scipy.linalg import block_diag
import pydare as dare

class Controller( object ):
    """Base, dummy controller class. Use derived classes."""
    def compute_control_input( self ):
        """Compute control input. This is a placeholder method: use derived classes."""
        raise NotImplemented( 'Call derived classes instead.')


class LQController( Controller ):
    def __init__ ( self, system, Q, R ):
        """An Infinite Horizon Linear Quadratic controller to 
        regulate to zero the state of the system.
               
        Consider the following LTI system model:
        
        .. math:: 
            \\mathbf{x}(k+1) = \\mathbf{A}\\mathbf{x}(k) + \\mathbf{B}\\mathbf{u}(k)
        
        and assume that the control objectives can be formulated as 
        the minimization of a quadratic cost function:
        
        .. math ::
            J(x(0), U) = \\sum_{i=0}^\\infty \\big( \\mathbf{x}(i)^T \\mathbf{Q} \\mathbf{x}(i) + \\mathbf{u}(i)^T \\mathbf{R} \\mathbf{u}(i) \\big )
            
        where :math:`\\mathbf{Q}>0` and :math:`\\mathbf{R}>0` are the state and input 
        weighting matrices, design parameters of the control problem.
        
        It can be shown that the optimal control input is a static state feedback:
        
        .. math::
            \\mathbf{u}^O(k) = - \\mathbf{K} \\mathbf{x}(k)
        
        where:
        
        .. math::
            \\mathbf{K} = \\big(\\mathbf{R}+\\mathbf{B}^T\\mathbf{P}\\mathbf{B}\\big)^{-1}\\mathbf{B}^T\\mathbf{P}\\mathbf{A}
            
        and :math:`\\mathbf{P}>0` is the positive definite solution of the Algebraic 
        Riccati Equation:
        
        .. math::
            \\mathbf{P} = \\mathbf{A}^T\\mathbf{P}\\mathbf{A} + \\mathbf{Q} - \\mathbf{A}^T \\mathbf{P}\\mathbf{B} (\\mathbf{R}+\\mathbf{B}^T\\mathbf{P}\\mathbf{B})^{-1} \\mathbf{B}^T\\mathbf{P}\\mathbf{A}
            
            
        Parameters
        ----------
        system : an instance of :py:class:`mpc.systems.DtLTISystem`, or one of its
                derived classes. This is the linear system which has 
                to be controlled.
                
        Q : numpy.matrix
            the state weigthing matrix. Must be positive definite
            and with shape ``(n_states, n_states)``.
            
        R : numpy.matrix
            the input weigthing matrix. Must be positive definite.
            and with shape ``(n_inputs, n_inputs)``.
            
        
        Attributes
        ----------
        K : numpy.matrix 
            the state feedback gain matrix
        """
        # make two local variables
        Ql = np.matrix( Q )
        Rl = np.matrix( R )
        
        # solve Algebraic Riccati Equation
        P = dare.DareSolver( system.A, system.B, Ql, Rl ).solve_direct()
        
        # create static state feedback matrix
        self.K  = (Rl + system.B.T * P * system.B).I * (system.B.T * P * system.A)
        
    def compute_control_input( self, x ):
        """Compute static state feedback. 
        
        Parameters
        ----------
        x : numpy.matrix, shape = ``(n_states, 1)``.
            the state column vector.
        
        Returns
        -------
        u : numpy.matrix, shape = ``(n_inputs, 1)``.
            an input vector which is fed back to the system.
            
        Notes
        -----
        The control move is computed as:
        
        .. math::
            
            u = - K x
        
        """
        
        return - np.dot( self.K, x)
