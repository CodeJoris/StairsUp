import numpy as np
from ins_tools.util import *
from EKF import Localizer

class INS():
    def __init__(self, imudata, has_gravity, sigma_a=0.01, sigma_w=0.1*np.pi/180, T=1.0/125, dt=None):
        self.config = {
            "sigma_a": sigma_a,
            "sigma_w": sigma_w,
            "g": 9.8029,
            "T": T,
            "dt": dt,
            "has_gravity": has_gravity
        }
        self.imudata = imudata
        self.sigma_a = self.config["sigma_a"]
        self.sigma_w = self.config["sigma_w"]
        self.var_a = np.power(self.sigma_a,2)
        self.config["var_a"] = self.var_a
        self.var_w = np.power(self.sigma_w,2)
        self.config["var_w"] = self.var_w
        self.g = self.config["g"]
        self.T = self.config["T"]
        ##process noise in body frame
        self.sigma_acc = 0.5*np.ones((1,3))
        self.var_acc = np.power(self.sigma_acc,2)
        self.sigma_gyro = 0.5*np.ones((1,3))*np.pi/180
        self.var_gyro = np.power(self.sigma_gyro,2)
    
        self.Q = np.zeros((6,6))  ##process noise covariance matrix Q
        self.Q[0:3,0:3] = self.var_acc*np.identity(3)
        self.Q[3:6,3:6] = self.var_gyro*np.identity(3)
        self.config["Q"] = self.Q
        
        self.sigma_vel = 0.01 #0.01 default
        self.R = np.zeros((3,3))
        self.R[0:3,0:3] = np.power(self.sigma_vel,2)*np.identity(3)   ##measurement noise, 0.01 default
        self.config["R"] = self.R
        
        self.H = np.zeros((3,9))
        self.H[0:3,3:6] = np.identity(3)
        self.config["H"]= self.H        
        
        self.Localizer = Localizer(self.config, imudata)
    
