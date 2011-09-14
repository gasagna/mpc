"""This module provide some controller classes."""

__author__ = 'Davide Lasagna, Politecnico di Torino Dipartimento di Ingegneria Aerospaziale. <davide.lasagna@polito.it>'
__date__ = '26/07/2011'


import numpy as np
from scipy.linalg import block_diag

import pydare as dare

__all__ = ['Controller', 'LQController']


class Controller( object ):
    """A Controller is an object with a single public method, ``compute_control_input``,
    which is responsible of returning the appropriate control input, based on the
    system's state, which is given as argument to the method.
    
    This is the base class from which all other derived class inherit. You should 
    use them instead of this one. For consistency, if you want to create your custom 
    Controller class you may want to inherit from this class.
    """
    def compute_control_input( self, x ):
        raise NotImplemented( 'Call derived classes instead.')


class LQController( Controller ):
    def __init__ ( self, system, Q, R ):
        """An Infinite Horizon Linear Quadratic controller (IHLQ).
        
        
        Consider the following LTI system model:
        
        .. math:: 
            x(k+1) = Ax(k)+Bu(k)
        
        and assume that the control objectives can be formulated as 
        the minization of a quadratic cost function:
        
        .. math ::
            J(x(0), U) = \sum_{i=0}^\infty \big( x(i)^T Q x(i) + u(i)^T R u(i) \big )
            
        where :math:`Q>0` and :math:`R>0` are the state and input 
        weighting matrices, design parameters of the control problem.
        
        It can be shown that the optimal control input is a static state feedback:
        
        .. math::
            u^O(k) = - K x(k)
        
        where:
        
        .. math::
            K = \big(R+B^TPB)^{-1}B^TPA
            
        and :math:`P>0` is the positive definite solution of the Algebraic 
        Riccati Equation:
        
        .. math::
            P = A^TPA + Q - A^T PB (R+B^TPB)^{-1} B^TPA
            
            
        Parameters
        ----------
        system : an instance of mpc.systems.DtLTISystem, or one of its
                derived classes. This is the linear system which has 
                to be controlled.
                
        Q : np.matrix
            the state weigthing matrix. Must be positive definite
            and with shape ``(n_states, n_states)``.
            
        R : np.matrix
            the input weigthing matrix. Must be positive definite.
            and with shape ``(n_inputs, n_inputs)``.
        
        """
        
        
        # make two local variables
        Ql = np.matrix( Q )
        Rl = np.matrix( R )
        
        # solve Algebraic Riccati Equation
        P = dare.DareSolver( system.A, system.B, Ql, Rl ).solve_slycot()
        
        # create static state feedback matrix
        self.K  = (Rl + system.B.T * P * system.B).I * (system.B.T * P * system.A)
        
    def compute_control_input( self, x ):
        """Compute static state feedback. 
        
        Parameters
        ----------
        x : np.matrix, shape = ``(n_states, 1)``.
            the state column vector.
        
        Returns
        -------
        u : np.matrix, shape = ``(n_inputs, 1)``.
            an input vector whic is fed back to the system.
            
        Notes
        -----
        The control move is computed as:
        
        ..math::
         u = - K x
        
        """
        
        return - np.dot( self.K, x)


class MPController( Controller ):
    """The following linear, discrete-time. time invariant model is used
    ..math::
        x(k+1) = A x(k) + B u(k)
        y(k) = C x(k)
        
    we want to design a predictive controller in order to obtain the 
    system state regulation to the oigin in the presence of:
    * input saturation constraints,
    * linear state constarints.
    
    For the control problem considered it can be shown that the MPC 
    controller is obtained by solving a Quadratic Programme (QP) problem
    at each sampling time. 
    
    The control objectives can be taken into account through the 
    following cost function.
    
    
    Consider the following quadratic cost function:
    
    ..math:: 
        J(x(k|k), U(k)) = \sum_{i=0}^{H_p-1} ||x(k+1|k)||^{2}_Q + ||u(k+1|K)||^{2}_R
        
    with:
    
    ..math::
        U(k) = [u(k|k) u(k+1|k) ... u(k+Hc-1|k)]^T
        
    and 
    ..math::
        H_p=H_c; \, Q=Q^T>=0; \, R=R^T>=0
        
    """
    def __init__ ( self, system, Q, R, Hp, Hc ):

        
        Ac = np.vstack( [system.A**i for i in range(1, Hp+1)] ) 
        
        rows = []
        for i in range(Hc):
            row = np.hstack( [ (system.A**(i-j) )*system.B if j<=i else np.zeros_like(system.B) for j in range(Hc)]  )
            rows.append(row)
            
        Bc = np.vstack(rows)
        
        Qc = block_diag( *(Q for i in range(Hp)) )
        Rc = block_diag( *(R for i in range(Hp)) )
        
        H = 2*( Bc.T*Qc*Bc + Rc )
        F = 2*Bc.T*Qc*Ac
        
        self.K = np.linalg.inv(H) * F
        
        
    def compute_control_input( self, x ):
        return - np.dot( self.K[0],  x)
