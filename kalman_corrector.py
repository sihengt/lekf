# Use https://github.com/pypose/pypose/blob/yuheng_cov_pro/examples/imu/imu_corrector.py as a guide to write this.

import torch
from torch import nn
from kalman_filter import KalmanFilter as kf

class KalmanCorrector(nn.module):
    # TODO: perhaps create variables for first last layer, and NN layers instead.
    def __init__(self, size_list=[6, 64, 128, 128, 128, 6]):
        super().__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)
        self.imu = kf.IMUPreintegrator(reset=True, prop_cov=False)
    
    def forward(self, data):
        pass

def train():
    pass
def test():
    pass

if __name__ == "__main__":
    pass