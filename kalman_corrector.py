# Use https://github.com/pypose/pypose/blob/yuheng_cov_pro/examples/imu/imu_corrector.py as a guide to write this.

import torch
from torch import nn
from kalman_filter import KalmanFilter as kf

class KalmanCorrector(nn.module):
    # TODO: perhaps create variables for first & last layer, and NN layers instead.
    # If assuming state equation covariance to be the identity matrix, you only have to predict n_state terms.
    def __init__(self, n_states, size_list=[6, 64, 128, 128, 128, 6]):
        super().__init__()
        self.n_states = n_states
        assert(self.n_states == size_list[0] and self.n_states == size_list[-1])
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)
        self.kf = kf.IMUPreintegrator(reset=True, prop_cov=False)
    
    def forward(self, data):
        # Assumes correction output.
        x = data['state']
        u = data['control']
        z = data['z']
        F = data['F'] # State transition function
        B = data['B'] # Control transition function
        H = data['H'] # Measurement function
        R = data['R'] # Measurement noise
        Q = data['Q'] # Process noise

        correction_output = self.net(x) # TODO: insert state here, get correction from here.
        for i in range(self.n_states):
            Q[i][i] + correction_output[i]
        
        return self.kf.forward(u, z, B=B, F=F, Q=Q, R=R, H=H)

def train():
    pass

def test():
    pass

if __name__ == "__main__":
    pass