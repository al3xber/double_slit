# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:25:23 2022

@author: Alexander
"""
import double_slit as ds
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ds_para:
    """
    Class for the double slit parameters.
    
    ds_para stands for double_slit_parameters
    """
    intensity: float
    sx: float
    x0: float
    offset: float
    a2: float
    a3: float
    phase: float
    
    def to_array(self):
        """
        make intuitively a ds_para object to a numpy array
        """
        return np.array([self.intensity,self.sx,self.x0,self.offset,self.a2,self.a3,self.phase])



def to_minimize(x, X=None, y=None):
    """
    Needed function, to minimize in the following using scipy.optimize.least_squares 
    
    Args:
        x:   numpy array with shape (7,) containing: (intensity,sx,x0,offset,a2,a3,phase)
        X:   numpy array with shape (n,) containing x-values
        y:   numpy array with shape (n,) containing corresponding y-values
    """
    return ds.double_slit(X, intensity=x[0], sx=x[1], x0=x[2], offset=x[3],
                          a2=x[4], a3=x[5], phase=x[6])-y

def jac(x, X=None, y=None):
    """
    Needed function, to compute jacobian scipy.optimize.least_squares 
    
    Args:
        x:   numpy array with shape (7,) containing: (intensity,sx,x0,offset,a2,a3,phase)
        X:   numpy array with shape (n,) containing x-values
        y:   numpy array with shape (n,) containing corresponding y-values
    """
    dy_dx0, dy_dI, dy_dxs, dy_dy0, dy_dphase, dy_da2, dy_da3 = ds.calc_derivatives(X, intensity=x[0], sx=x[1], x0=x[2], offset=x[3],
                          a2=x[4], a3=x[5], phase=x[6])                         
    
    return np.array([dy_dI,dy_dxs,dy_dx0,dy_dy0,dy_da2,dy_da3,dy_dphase]).T

class double_slit_reg():
    """Linear perceptron classifier.
    Read more in the :ref:`User Guide <perceptron>`.
    
    Parameters
    ----------
    x0_pred : ds_para object,
        Initial guess for prediction
        
    Attributes
    ----------
    x0_pred : ds_para object,
        Either initial guess or updated guess after fitting
    cost: float 
        Mean absolute error of fit 
    x:  ds_para object,
        solution after fit
    ----------
    """

    def __init__(
        self,
        x0_pred=ds_para(2.6,3.8,0.2,0.2,2.0,2.2,1.5 * np.pi)
    ):
        self.x0_pred=x0_pred
        self.cost=None
        self.x=None
        
    def fit(self, X, y):
        """
        Args:
        X:   numpy array with shape (n,) containing x-values
        y:   numpy array with shape (n,) containing corresponding y-values
        
        Saves attributes: x0_pred, cost, x
        """
        args=(X,y)
        res = least_squares(to_minimize,self.x0_pred.to_array(),jac,args=args,
                           bounds=([0,0,-np.inf,-np.inf,-np.inf,-np.inf,0], np.inf))
        self.x0_pred=ds_para(*res.x.tolist())
        self.cost=res.cost
        self.x=ds_para(*res.x.tolist())
        
    def predict(self,X):
        """
        Args:
        X:   numpy array with shape (n,) containing x-values
        
        Returns 
             numpy array with shape (n,) containing corresponding y-values
        """
        if self.x is None:
            raise Exception("Please train first, before predicting")
        return ds.double_slit(X, intensity=self.x.intensity, sx=self.x.sx, 
                      x0=self.x.x0, offset=self.x.offset, a2=self.x.a2, 
                      a3=self.x.a3, phase=self.x.phase)
        
        
        
    def plot_result(self,X,y,true_input=None):
        """
        Args:
        X:   numpy array with shape (n,) containing x-values
        y:   numpy array with shape (n,) containing corresponding y-values
        
        true_input: optional, ds_para object 
                    If available one can compare the true and predicted intensity distribution
        Plots: Datapoints, prediction and potentially true curve
        """
        z = np.linspace(-3, 3, num=10*60+1)
        plt.plot(z,self.predict(z),
                 label = "prediction",alpha=0.5)
        if true_input!=None:
            plt.plot(z,ds.double_slit(z, intensity=true_input.intensity, sx=true_input.sx, 
                          x0=true_input.x0, offset=true_input.offset, a2=true_input.a2, 
                          a3=true_input.a3, phase=true_input.phase),
                label = "true",alpha=0.5)
        plt.scatter(X,y,s=4)
        plt.legend()
        #plt.xlim([1,2])
        
__all__=["ds_para","double_slit_reg"]
