import numpy as np
from pyhd.sig import acorr
from scipy.linalg import toeplitz
from numpy.linalg import solve
from pyhd.base import overlap_array


class AutoRegressiveModel(object):
    """ A class to create an autoregressive model from data.
    
    An autoregressive model is of the form:
    
    .. math::
            x_{k}  = \\sum_{i=1}^{p} a_i x_{k-i} + e_k
            
    where :math:`p` is the model order,  :math:`a_i`  are the
    autoregressive coefficients and :math:`e_t` is the prediction
    error. The errors are assumed Gaussian with zero mean
    
    Attributes
    ----------
    data : np.ndarray
        an array for the data
        
    order : int
        the order of the model
        
    coeffs : np.ndarray
        the coefficients of the autoregressive model
    """
    def __init__ ( self, data, order  ):
        """ Create an Auto-Regressive model from data.
        
        Parameters
        ----------
        data : np.ndarray
            an array for the data
            
        order : int
            the order of the model
        """
        # set some attributes
        self.data = data
        self.order = order
        
        # compute signal autocorrelation 
        R = acorr( data )
        
        # create Toepliz matrix of shape p*p
        T = toeplitz( R[:order] )
        
        # solve for coefficient of the AR model
        self.coeffs = solve( T, R[1:order+1] ) 
        
    def process_noise_variance( self, n ):
        """Estimate the process noise variance from the prediction error.
        
        .. math::
            {\\sigma_e}^2 = var( \\mathbf{e} ) = var( \\mathbf{x} - \\mathbf{\\hat{x}})
        
        Parameters
        ----------
        n : int 
            the lenght of the error vector from which the noise variance
            will be computed.
        
        Returns
        -------
        evar : float
            the variance of the process noise 
        """
        xcap = self.estimate( n )
        e = overlap_array( self.data, len(self.coeffs)+1, len(self.coeffs) )[:n,-1] - xcap
        return e.var()
        
    def estimate( self, n ):
        """Estimate step.
        
        .. math::
            \hat{x}_{k}  = \\sum_{i=1}^{p} a_i x_{k-i}
        
        Parameters
        ----------
        n : int 
            the lenght of the error vector from which the noise variance
            will be computed. Should be the largest allowed from your system.
        """
        X = overlap_array( self.data, len(self.coeffs), len(self.coeffs)-1 )[:n]
        return np.dot( X, self.coeffs.T ) 
