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
    Dataclass for the double-slit parameters.

    ds_para stands for double-slit parameters
    """
    intensity: float
    sx: float
    x0: float
    offset: float
    visibility: float
    a3: float
    phase: float

    def to_array(self):
        """
        make intuitively a ds_para object to a numpy array
        """
        return np.array([self.intensity,self.sx,self.x0,self.offset,self.visibility,self.a3,self.phase])



def to_minimize(x, X=None, y=None):
    """
    Helping function, to minimize in the following using scipy.optimize.least_squares

    Args:
        x:   numpy array with shape (7,) containing: (intensity,sx,x0,offset,visibility,a3,phase)
        X:   numpy array with shape (n,) containing x-values
        y:   numpy array with shape (n,) containing corresponding y-values
    """
    return ds.double_slit(X, intensity=x[0], sx=x[1], x0=x[2], offset=x[3],
                          visibility=x[4], a3=x[5], phase=x[6])-y

def jac(x, X=None, y=None):
    """
    Helping function, to compute jacobian scipy.optimize.least_squares

    Args:
        x:   numpy array with shape (7,) containing: (intensity,sx,x0,offset,visibility,a3,phase)
        X:   numpy array with shape (n,) containing x-values
        y:   numpy array with shape (n,) containing corresponding y-values
    """
    dy_dx0, dy_dI, dy_dxs, dy_dy0, dy_dphase, dy_dvisibility, dy_da3 = ds.calc_derivatives(X,
                        intensity=x[0], sx=x[1], x0=x[2], offset=x[3], visibility=x[4], a3=x[5], phase=x[6])

    return np.array([dy_dI,dy_dxs,dy_dx0,dy_dy0,dy_dvisibility,dy_da3,dy_dphase]).T

# ds_para(intensity=2105.7825767721392, sx=0.012717336182275022, x0=199.5593207697112,
#                offset=172.87709664672613, visibility=-0.9532526623852442, a3=0.19936603574497574, phase=0.1926458678720715),
# ds_para(intensity=2179.3116820189953, sx=0.012622553939985863, x0=195.906792595373,
#               offset=164.0075019994732, visibility=-0.32285965987255955, a3=0.1981786243155078, phase=12.520980153346958),
#     ds_para(intensity=2390.8345717461007, sx=0.012738042816677926, x0=206.0289013643262,
# offset=171.41599809148641, visibility=-0.8418428679062707,a3=0.1991244454392781, phase=0)

ds_para_default = [
    ds_para(intensity=7005,
            sx=2712,
            x0=2.8e-3,
            offset=159,
            visibility=-0.868390150362010,
            a3=42693,
            phase=0)
]

class double_slit_reg():
    """Double Slit Regressor.
    For more intuition view files in the example folder.

    Parameters
    ----------
    x0_pred : ds_para object, optional
        Initial guess for prediction

    Attributes
    ----------
    cost: float
        Summed squared error of fit
    x:  ds_para object,
        solution after fit
    fit_success: boolean
                 True if fit was successfull else False
    res: scipy class object
         result of gradient descent fit
    ----------
    """

    def __init__(self,
        x0_pred = ds_para(intensity=2390, sx=0.012738, x0=206, offset=171.42, visibility=-0.84,a3=0.1991244454392781, phase=0)):

        self.cost = None
        self.x = x0_pred
        self.fit_success = False
        self.res = None

    def fit(self, X, y):
        """
        Args:
        X:   numpy array with shape (n,) containing x-values
        y:   numpy array with shape (n,) containing corresponding y-values

        Saves attributes: cost, x
        """

        x0_values_list = [self.x,  #the following ds_para objects are all good initial guesses, obtained after training
                                   #a double slit regression task
        ] + ds_para_default

        error_list = []  #we remember the errors and parameters of the (bad) predictions, s.t. we can pick the best one,
        para_list = []   #in case all predictions are bad
        res_list = []

        self.fit_success = False

        for x0_value in x0_values_list: #go through all elements of the initial guesses list

            res = least_squares(to_minimize, x0_value.to_array(), jac, args=(X,y),  #least_squares by scipy.optimize
                           bounds=([0,0,-np.inf,-np.inf,-np.inf,-np.inf,0], np.inf))

            if res.cost < 1300*len(X) and abs(res.x.tolist()[5])<1:  #hand-picked border for a good prediction. 2nd condition
                                                                     #checks if the solution is oscillating too strong
                self.cost = res.cost
                self.x = ds_para(*res.x.tolist())
                self.fit_success = True
                self.res = res
                break  #in case we found a good prediction, remember the parameters and end fitting
            elif abs(res.x.tolist()[5])<10:  #check if the solution is oscillating too strong, if so ignore
                error_list.append(res.cost)
                para_list.append(res.x)
                res_list.append(res)
            else:
                error_list.append(res.cost)
                para_list.append(res.x)
                res_list.append(res)

        if not self.fit_success:  #in case we were not able to find a good prediction, take the best we have seen.
            # print(error_list)
            best_idx = np.argmin(error_list)
            self.cost = error_list[best_idx]
            self.x = ds_para(*para_list[best_idx].tolist())
            self.res = res_list[best_idx]

    def predict(self,X):
        """
        Predicts to given x-values the corresponding y-values, using the parameters saved in self.x. Make sure to
        train before predicting!

        Args:
        X:   numpy array with shape (n,) containing x-values

        Returns
             numpy array with shape (n,) containing corresponding y-values
        """
        if self.cost is None:
            raise Exception("Predicting is only possible after training")
        return ds.double_slit(X, intensity=self.x.intensity, sx=self.x.sx,
                      x0=self.x.x0, offset=self.x.offset, visibility=self.x.visibility,
                      a3=self.x.a3, phase=self.x.phase)



    def plot_result(self,X,y,x0_input=None,plot_bounds=(0,450)):
        """
        Args:
        X:   numpy array with shape (n,) containing x-values
        y:   numpy array with shape (n,) containing corresponding y-values

        x0_input: optional, ds_para object
                    If available one can compare the x0 input and predicted intensity distribution
        plot_bounds: optional, tuple/list object
                    sets the plot bounds
        Plots: Datapoints, prediction and potentially true curve
        """

        z = np.linspace(plot_bounds[0], plot_bounds[1], num=10*600+1)  #plotting space

        plt.plot(z,self.predict(z),  #plot prediction
                 label = "prediction",alpha=0.5)
        if x0_input!=None:
            plt.plot(z,ds.double_slit(z, intensity=x0_input.intensity, sx=x0_input.sx,  #plot x0_input, if given
                          x0=x0_input.x0, offset=x0_input.offset, visibility=x0_input.visibility,
                          a3=x0_input.a3, phase=x0_input.phase),
                label = "$x_0$ input",alpha=0.5)
        plt.scatter(X,y,s=4)  #scatter the data points
        plt.legend()
        plt.show()

__all__=["ds_para","double_slit_reg"]
