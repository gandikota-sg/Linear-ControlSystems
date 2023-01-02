# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        self.lr = 0.82
        self.lf = 1.18
        self.Ca = 20000
        self.Iz = 3004.5
        self.m = 1000
        self.g = 9.81
        self.preve2 = 0

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

        # Find the closest node to the vehicle
        _, node = closestNode(X, Y, trajectory)

        # Choose a node that is ahead of our current node based on index
        forwardIndex = 100

        # Determine desired heading angle and e1 using two nodes - one ahead, and one closest
        # We use a try-except so we don't attempt to grab an index that is out of scope

        # To define our error-based states, we use definitions from documentation.
        # Please see page 34 of Rajamani Rajesh's book "Vehicle Dynamics and Control", which
        # is available online through the CMU library, for more information.
        # It is important to note that numerical derivatives of e1 and e2 will also work well.
        try:
            psiDesired = np.arctan2(trajectory[node+forwardIndex,1]-trajectory[node,1], \
                                    trajectory[node+forwardIndex,0]-trajectory[node,0])
            e1 =  (Y - trajectory[node+forwardIndex,1])*np.cos(psiDesired) - \
                  (X - trajectory[node+forwardIndex,0])*np.sin(psiDesired)

        except:
            psiDesired = np.arctan2(trajectory[-1,1]-trajectory[node,1], \
                                    trajectory[-1,0]-trajectory[node,0])
            e1 =  (Y - trajectory[-1,1])*np.cos(psiDesired) - \
                  (X - trajectory[-1,0])*np.sin(psiDesired)

        e1dot = ydot + xdot*wrapToPi(psi - psiDesired)
        e2 = wrapToPi(psi - psiDesired)
        # e2dot = psidot # This definition would be psidot - psidotDesired if calculated from curvature
        e2dot = (e2 - self.preve2)/delT
        self.preve2 = e2
        # Assemble error-based states into array
        states = np.array([e1,e1dot,e2,e2dot])
        Q = np.array([[10, 0, 0, 0],
                      [0, 0.1, 0, 0],
                      [0, 0, 0.1, 0],
                      [0, 0, 0, 0.01]])
        R = 75
        N = 30
        #Discretize Sys
        continous_sys = signal.StateSpace(A, B, C, D)
        discrete_sys = continous_sys.to_discrete(delT)
        S = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        K = [0, 0, 0, 0]
        for i in range(N,1,-1):
            new_K = np.matrix(-1 * linalg.inv(R + np.transpose(discrete_sys.B) @ S @ discrete_sys.B) @ (np.transpose(discrete_sys.B) @ S @ discrete_sys.A))
            K = np.vstack((K, new_K))
            S = np.matrix(np.transpose(discrete_sys.A + discrete_sys.B @ new_K)) @ S @ np.matrix(discrete_sys.A + discrete_sys.B @ new_K) + Q + (np.transpose(new_K) * R) @ new_K
        delta = wrapToPi((K[-1] @ states)[0,0])

        # ---------------|Longitudinal Controller|-------------------------

        kp = 50
        ki = 30
        kd = 10

        # Reference value for PID to tune to
        desiredVelocity = 7

        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError

        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta