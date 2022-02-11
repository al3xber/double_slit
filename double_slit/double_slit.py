"""
Created on Tue Feb  1 15:35:20 2022

@author: Alexander
"""

import numpy as np
from scipy.special import sinc, cotdg


def double_slit_orig(x, a0, a1, a2, a3, a4, a5, a6):
    """
    Original calculation of the distribution of light intensity

    Parameters
    ----------
    x : numpy array or float
    a0 : float
    a1 : float
    a2 : float
    a3 : float
    a4 : float
    a5 : float
    a6 : float

    Returns
    -------
    II : numpy array / float.

    """
    II = (
        a0
        * (np.sin(a1 * (x - a4)) / (a1 * (x - a4))) ** 2
        / 2
        * (1 + a2 * np.cos(a3 * (x - a4) + a5))
        + a6
    )
    for k in np.arange(np.size(II)):
        if np.size(II) > 1 and II[k] != II[k]:    
            II[k] = a0 + a6
    return II


def double_slit(x, *, sx, intensity, a2, a3, x0, phase, offset):
    """
    Numericaly more stable calculation of light intensity
    Args:
        x:   running parameter
        x0:  center of x
        scale: scaling of x axis
        
        a1:
        a2:
        a3:
        x0:    
    """
    
    if x0!=0:
        dx = x0*(x/x0 - 1)   #equivalent to x-x0
    else: # case x0==0
        dx = x 
    
    sdx = sx * dx
    
    sinc_val = sinc(sdx/np.pi)             #sinc(x)= sin(x*pi)/(x*pi) 
    term1 = 1 / 2 * sinc_val**2
    term2_inner = a2 * np.cos(a3 * dx + phase) 
    term2 = 1 + term2_inner
    
    y = intensity * term1 * term2 + offset
    return y
    
    
__all__=["double_slit_orig","double_slit"]
