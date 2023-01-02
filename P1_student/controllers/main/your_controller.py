# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 0.82
        self.lf = 1.18
        self.Ca = 20000
        self.Iz = 3004.5
        self.m = 1000
        self.g = 9.81
       
        #Error
        self.psi_cumm_error = 0
        self.psi_prev_error = 0
        
        self.vel_cumm_error = 0
        self.vel_prev_error = 0
     
         #Look Ahead
        self.dist_forward = 50
     
     
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 
        
        #--------------------------|Look Ahead|-----------------------
        dist_forward = self.dist_forward
        _, close_pt = closestNode(X, Y, trajectory)
        #print(trajectory)
        dist_monitored = close_pt + dist_forward
        if (dist_monitored >= trajectory.shape[0]):
            dist_forward = 0
        #print(dist_monitored)
        expected_X = trajectory[close_pt + dist_forward, 0]
        expected_Y = trajectory[close_pt + dist_forward, 1]
        expected_psi = np.arctan2(expected_Y - Y, expected_X - X)

        # ---------------|Lateral Controller|-------------------------
        kp_lateral = 5 
        ki_lateral = 0.001
        kd_lateral = 0.001
        
        psi_error = wrapToPi(expected_psi - psi) #Mark between -pi to pi 
        #Update Psi error PID
        psi_diff_error = (psi_error - self.psi_prev_error) / delT 
        self.psi_cumm_error = self.psi_cumm_error + psi_error * delT
        self.psi_prev_error = psi_error
        delta = kp_lateral * psi_error + ki_lateral * self.psi_cumm_error + kd_lateral * psi_diff_error

        #pi/6 check
        if (delta < -3.1416 / 6):
            delta = -3.1415 / 6
        elif (delta > 3.1416 / 6):
            delta = 3.1416 / 6

        # ---------------|Longitudinal Controller|-------------------------
        
        kp_longitudinal = 2 
        ki_longitudinal = 0.001
        kd_longitudinal = 0.001
        
        vel_error = (np.power(np.power(expected_X - X, 2) + np.power(expected_Y - Y, 2), 0.5)) / delT
        
        #Update vel error PID 
        vel_diff_error = (vel_error - self.vel_prev_error) / delT
        self.vel_cumm_error = self.vel_cumm_error + vel_error * delT
        self.vel_prev_error = vel_error
        F = kp_longitudinal * vel_error + ki_longitudinal * self.vel_cumm_error + kd_longitudinal * vel_diff_error
       
        #Force range check
        if (F > 16000):
            F = 16000
        elif(F < 0):
            F = 0

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
