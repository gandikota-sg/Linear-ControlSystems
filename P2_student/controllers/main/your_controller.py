# Fill in the respective functions to implement the full-state feedback controller

# Import libraries

import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

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

        #Error
        self.psi_cumm_error = 0
        self.psi_prev_error = 0 

        self.vel_cumm_error = 0
        self.vel_prev_error = 0
        
         #Look Ahead
        self.dist_forward = 50
        expected_V = 15;

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
        A = np.array([
            [0,1,0,0],
            [0,-4*Ca/(m*xdot), 4*Ca/m, -2*Ca*(lf-lr)/(m*xdot)],
            [0,0,0,1],
            [0,-2*Ca*(lf-lr)/(Iz*xdot), 2*Ca*(lf-lr)/Iz,
             -2*Ca*(lf**2+lr**2)/(Iz*xdot)]
            ])
        B = np.array([[0],[2*Ca/m],[0],[2*Ca*lf/Iz]])
        C = np.eye(4)
        D = np.zeros((4,1))

        poles = np.array([-15,-25,-1,0])
        #Scipy Package Pole Placing
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.place_poles.html
        K = signal.place_poles(A, B, poles).gain_matrix

        e1 = (np.power(np.power(expected_X - X, 2) + np.power(expected_Y - Y, 2), 0.5))
        e2 = wrapToPi(psi - expected_psi)
        e1_dot =  xdot*wrapToPi(psi - expected_psi)
        e2_dot = psidot 
        delta = wrapToPi(np.dot(-K,np.hstack((e1, e1_dot, e2, e2_dot)).reshape(4,1))[0,0])

        # ---------------|Longitudinal Controller|-------------------------

        kp_longitudinal = 29
        ki_longitudinal = 0.001
        kd_longitudinal = 0.001
        
        vel_error = expected_V - xdot
        #Update vel error PID 
        #https://pidexplained.com/pid-controller-explained/
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


