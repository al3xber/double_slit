# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:37:20 2022

@author: Alexander
"""
import numpy as np
import double_slit
import sympy
from scipy.special import sinc

def test_signature():
    """
    Check if the functions matches the expected signature
    Returns
    -------
    None.

    """
    double_slit.double_slit(0, intensity=1, sx=10, x0=0, offset=0, visibility=2, a3=2, phase=1.5 * np.pi)
    
def test_scipy_and_sympy_sinc():
    """
    Check the relation between scipy.special and sympy sinc 
    Further check scipy.special.sinc(x)=sin(x*pi)/(x*pi)
    While         sympy.sinc(x)=sin(x)/x 
    and check if  sympy.sinc(0)==scipy.special.sinc(0)== 1
    -------
    None.

    """
    x = sympy.Symbol('x', real=True)
    y = sympy.sinc(x)
    assert np.abs(y.evalf(subs={x:3})-sinc(3/np.pi))<10**(-17)
    assert sinc(1/2)==2/(np.pi)
    assert y.evalf(subs={x:0})==1 and sinc(0)==1
    
def test_offset():
    """
    Check if the offset parameter behaves as expected
    The offset parameter only changes the height of the distribution
    We first check that property
    The next check is to ensure that the derivative is equal to 1
    -------
    None.

    """
    z = np.linspace(-1, 1, num=10*60+1)     
    
    assert np.all(double_slit.double_slit(z, intensity=1, sx=10, x0=0, offset=1, visibility=2, a3=2, phase=1.5 * np.pi)  == \
           double_slit.double_slit(z, intensity=1, sx=10, x0=0, offset=0, visibility=2, a3=2, phase=1.5 * np.pi) + 1)
           
    assert np.all(double_slit.calc_derivatives(z, intensity=1, sx=10, x0=0.1, offset=0, visibility=2, a3=2, phase=1.5 * np.pi)[3] == np.ones_like(z))


