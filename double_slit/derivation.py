# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 20:33:55 2022

@author: Alexander
"""
import numpy as np
from scipy.special import sinc

def calc_derivatives(x, *, sx, intensity, visibility, a3, x0, phase, offset):
    """
    Calculate the partial derivatives of the intensity in the double-slit experiment. 
    See analysis folder for better understanding
    This derivative calculation is later used in the gradient descent parameter
    estimation.
    
    
    Args:
        x:   running parameter
        x0:  center of x    
        sx: scaling of x axis
        intensity: float
        visibility: float
        a3: float
        phase: float
        offset: float
    """
    if x0!=0:
        dx = x0*(x/x0 - 1)
    else: 
        dx = x
        
    sdx = sx * dx
    
    if phase != 0:
        cr = phase * (1 + a3 * dx/phase) 
    else:
        cr = phase + a3 * dx
        
    cr2 = visibility * np.cos(cr) + 1
    
    scr=np.sin(cr)
    ccr=np.cos(cr)
    sinc_val = sinc(sdx/np.pi)             #sinc(x)= sin(x*pi)/(x*pi) 
    
    term1 = (1 / 2) * sinc_val**2

    """
    Calculation of :
    x * cos x - sin * x
    -------------------
           x * x
    """
    
    terms = 1 / np.array([-3, 30, -840, 45360])           #parameters of taylor approximation
    result = np.zeros_like(sdx)
    t_x = sdx
    x2 = sdx**2
    for term in terms:
        result += term * t_x
        t_x = t_x * x2
        
        
    result[sdx==0]=0  #given by formula, see in analysis folder sympy derivative calculation 
    
    #---------------------------------------------------
    #Outside of the interval (-0.05,0.05) the taylor approximation is inexact (error > 10**-17)
    #therefor we use the cotangens and sinus approximation outside of the interval, see analysis.ipynb for more details
    #---------------------------------------------------
    
    mask = np.abs(sdx)>=0.05
    result[mask]=(1/np.tan(sdx[mask])-1/sdx[mask])*np.sin(sdx[mask])/sdx[mask]    
    #---------------------------------------------------
    #Why the following terms are calculated as they are, view analysis folder
    
    dy_dx0    = intensity*visibility*a3*scr*term1-intensity*cr2*sx*result*sinc_val 
    dy_dI     = cr2*term1
    dy_dxs    = intensity*cr2*dx*result*sinc_val
    dy_dy0    = np.ones_like(x)
    dy_dphase = -intensity*visibility*scr*term1
    dy_dvisibility    = intensity*ccr*term1
    dy_da3    = -intensity*visibility*dx*scr*term1

    return dy_dx0, dy_dI, dy_dxs, dy_dy0, dy_dphase, dy_dvisibility, dy_da3
    
    
__all__=["calc_derivatives"]
