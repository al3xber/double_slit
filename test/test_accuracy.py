# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:25:30 2022

@author: Alexander
"""
import sympy
import numpy as np
import double_slit as ds

#Definition of all needed symbols
x = sympy.Symbol('x', real=True)
y = sympy.Symbol('y', real=True)
x0 = sympy.Symbol('x_0', real=True)
sx = sympy.Symbol('s_x', real=True)
intensity = sympy.Symbol('I', real=True)
phase = sympy.Symbol(r'\varphi', real=True)
visibility = sympy.Symbol('vis', real=True)
a3 = sympy.Symbol('a_3', real=True)
offset = sympy.Symbol('y_0', real=True)
#-----------------------------------------------
#Setting up the distribution of intensity

dx = x - x0
sdx = sx * dx
    
sinc_val = sympy.sinc(sdx)
term1 = sympy.Rational(1, 2) * sinc_val**2
term2_inner = visibility * sympy.cos(a3 * dx + phase)
term2 = 1 + term2_inner 
    
y = intensity * term1 * term2 + offset #equals the final distribution
#------------------------------------------------
#Calculating derivatives

dy_dx0 = sympy.diff(y, x0)
dy_dI = sympy.diff(y, intensity)
dy_dxs = sympy.diff(y, sx)
dy_dy0 = sympy.diff(y, offset)
dy_dphase = sympy.diff(y, phase)
dy_dvisibility = sympy.diff(y, visibility)
dy_da3 = sympy.diff(y, a3)

derivs = [dy_dx0, dy_dI, dy_dxs, dy_dy0, dy_dphase, dy_dvisibility, dy_da3] #list of all derivatives



def test_intensity_calc():
    """
    Checks the function double_slit, which calculates of the intensity distrubition 
    Returns
    -------
    None.

    """
    z = np.linspace(-1, 1, num=10*6+1) #interval and number of points to check
    
    true_intensity = [y.evalf(subs={x: z_i,intensity: 1, sx:10, x0:0, offset:0, visibility:2, a3:2, phase:1.5 * np.pi}) for z_i in z]
                     #the function evalf takes a sympy expression and allows to insert values.
                     
    calc_intensity = ds.double_slit(z, intensity=1, sx=10, x0=0, offset=0, visibility=2, a3=2, phase=1.5 * np.pi)

    assert np.max(np.abs(true_intensity-calc_intensity))<10**(-15)


def test_derivative_calc():
    """
    Checks the function calc_derivatives out of derivatives.py, which calculates of the derivatives 
    Returns
    -------
    None.

    """
    z = np.linspace(-1, 1, num=10*6+1) #interval and number of points to check
    
    exact_derivs=[]
    for i in range(len(derivs)):
        exact_derivs.append([derivs[i].evalf(subs={x: z[j],intensity: 1,sx:10, x0:0.1, offset:0, visibility:2, a3:2, phase:1.5 * np.pi}) for j in range(len(z))])
    
    calc_derivs=ds.calc_derivatives(z, intensity=1, sx=10, x0=0.1, offset=0, visibility=2, a3=2, phase=1.5 * np.pi)
    
    for i in range(len(derivs)):
        assert np.max(np.abs(exact_derivs[i]-calc_derivs[i]))<10**(-13)
