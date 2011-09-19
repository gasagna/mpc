__author__ = 'Davide Lasagna, Politecnico di Torino Dipartimento di Ingegneria Aerospaziale. <davide.lasagna@polito.it>'
__date__ = '20/09/2011'
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

import numpy as np

class KalmanStateObserver( object ):
    def __init__ ( self, system, x0=None ):
        
        # set attribute
        self._A = system.A
        self._B = system.B
        self._C = system.C
        self._D = system.D
        self._Sw = system.Sw
        self._Sv = system.Sv
        
        # set initial condition for state estimate
        if x0 is None:
            self.xhat = system.x + np.matrix( np.random.multivariate_normal( np.zeros((system.n_states,)), system.Sw, 1 ).reshape(system.n_states,1) )
        else:
            # check initial condition
            x0 = np.asmatrix( x0 )
            if not x0.shape == ( self.n_states, 1 ):
                raise DtSystemError('wrong shape of initial state vector')
            self._xhat = x0
        
        # covariance matrix of the state estimate
        self.P = self._Sw
        
    def get_state_estimate( self, y, u ):
        
        #simulate system with state estimate at previous step
        self.xhat = self._A * self._xhat + self._B * u
        
        # form the innovation vector
        inn = y - self._C*self.xhat
        
        # compute the covariance of the innovation
        s = self._C*self.P*self._C.T + self._Sv
        
        # form the kalman gain matrix
        K = self._A*self.P*self._C.T * np.linalg.inv(s)
        
        # update state estimate
        self.xhat += K*inn
        
        # compute covariance of the estimation error
        self.P = self._A*self.P*self._A.T -  \
                 self._A*self.P*self._C.T * np.linalg.inv(s)*\
                 self._C*self.P*self._A.T + self._Sw
        
        # return state estimate
        return self.xhat
