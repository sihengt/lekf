# Adapted from Roger Labbe's FilterPy
# https://github.com/rlabbe/filterpy

import torch
import kf
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from collections import defaultdict
import inspect
import numpy as np

#### UTILITY FUNCTIONS ####
def order_by_derivative(Q, dim, block_size):
    """
    Given a matrix Q, ordered assuming state space
        [x y z x' y' z' x'' y'' z''...]

    return a reordered matrix assuming an ordering of
       [ x x' x'' y y' y'' z z' y'']

    This works for any covariance matrix or state transition function

    Parameters
    ----------
    Q : np.array, square
        The matrix to reorder

    dim : int >= 1

       number of independent state variables. 3 for x, y, z

    block_size : int >= 0
        Size of derivatives. Second derivative would be a block size of 3
        (x, x', x'')
    """

    N = dim * block_size

    D = torch.zeros((N, N))

    Q = torch.tensor(Q)
    for i, x in enumerate(Q.ravel()):
        f = torch.eye(block_size) * x
        ix, iy = (i // dim) * block_size, (i % dim) * block_size
        D[ix:ix+block_size, iy:iy+block_size] = f

    return D

def Q_discrete_white_noise(dim, dt=1., var=1., block_size=1, order_by_dim=True):
    """
    Returns the Q matrix for the Discrete Constant White Noise
    Model. dim may be either 2, 3, or 4 dt is the time step, and sigma
    is the variance in the noise.

    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.

    Parameters
    -----------

    dim : int (2, 3, or 4)
        dimension for Q, where the final dimension is (dim x dim)

    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

    var : float, default=1.0
        variance in the noise

    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.

    order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)

        [x x' x'' y y' y'']

        whereas `False` interleaves the dimensions

        [x y z x' y' z' x'' y'' z'']


    Examples
    --------
    >>> # constant velocity model in a 3D world with a 10 Hz update rate
    >>> Q_discrete_white_noise(2, dt=0.1, var=1., block_size=3)
    array([[0.000025, 0.0005  , 0.      , 0.      , 0.      , 0.      ],
           [0.0005  , 0.01    , 0.      , 0.      , 0.      , 0.      ],
           [0.      , 0.      , 0.000025, 0.0005  , 0.      , 0.      ],
           [0.      , 0.      , 0.0005  , 0.01    , 0.      , 0.      ],
           [0.      , 0.      , 0.      , 0.      , 0.000025, 0.0005  ],
           [0.      , 0.      , 0.      , 0.      , 0.0005  , 0.01    ]])

    References
    ----------

    Bar-Shalom. "Estimation with Applications To Tracking and Navigation".
    John Wiley & Sons, 2001. Page 274.
    """

    if dim not in [2, 3, 4]:
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[.25*dt**4, .5*dt**3],
             [ .5*dt**3,    dt**2]]
    elif dim == 3:
        Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
             [ .5*dt**3,    dt**2,       dt],
             [ .5*dt**2,       dt,        1]]
    elif dim == 4:
        Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
             [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
             [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
             [(dt**3)/6,  (dt**2)/2 ,  dt,        1.]]

    if order_by_dim:
        return block_diag(*[Q]*block_size) * var
    return order_by_derivative(torch.tensor(Q), dim, block_size) * var

#### SETUP ####
class PosSensor1(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]

    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        return [self.pos[0] + torch.randn() * self.noise_std,
                self.pos[1] + torch.randn() * self.noise_std]

def const_vel_filter(dt, x0=0, x_ndim=1, P_diag=(1., 1.), R_std=1., Q_var=0.0001):
    """ helper, constructs 1d, constant velocity filter"""
    f = kf.KalmanFilter(dim_x=2, dim_z=1)

    if x_ndim == 1:
        f.x = torch.tensor([x0, 0.])
    else:
        f.x = torch.tensor([[x0, 0.]]).T

    f.F = torch.tensor([[1., dt],
                    [0., 1.]])

    f.H = torch.tensor([[1., 0.]])
    f.P = torch.diag(P_diag)
    f.R = torch.eye(1) * (R_std**2)
    f.Q = Q_discrete_white_noise(2, dt, Q_var)

    return f

def const_vel_filter_2d(dt, x_ndim=1, P_diag=(1., 1, 1, 1), R_std=1.,
                        Q_var=0.0001):
    """ helper, constructs 1d, constant velocity filter"""

    kf = kf.KalmanFilter(dim_x=4, dim_z=2)

    kf.x = torch.tensor([[0., 0., 0., 0.]]).T
    kf.P *= torch.diag(P_diag)
    kf.F = torch.tensor([
        [1., dt, 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., dt],
        [0., 0., 0., 1.]
    ])

    kf.H = torch.tensor([
        [1., 0, 0, 0],
        [0., 0, 1, 0]
    ])

    kf.R *= torch.eye(2) * (R_std**2)
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)
    kf.Q = block_diag(q, q)

    return kf

def test_noisy_1d():
    f = kf.KalmanFilter(dim_x=2, dim_z=1)

    f.x = torch.tensor([
        [2.],
        [0.]
        ])              # initial state (location and velocity)

    f.F = torch.tensor([
        [1., 1.],
        [0., 1.]
    ])                  # state transition matrix

    f.H = torch.tensor([
        [1., 0.]
    ])                  # Measurement function
    f.P *= 1000.        # covariance matrix
    f.R = torch.tensor([[5.]])       # state uncertainty
    f.Q = torch.tensor([[0.0001]])  # process uncertainty

    measurements = []
    results = []

    zs = []
    for t in range(100):
        # create measurement = t plus white noise
        z = t + torch.randn(1)*20
        zs.append(z)

        # perform kalman filtering
        f.update(z)
        f.predict()

        # save data
        results.append(f.x[0, 0])
        measurements.append(z)

    # plot data
    if DO_PLOT:
        p1, = plt.plot(measurements, 'r', alpha=0.5)
        p2, = plt.plot(results, 'b')
        p3, = plt.plot([0, 100], [0, 100], 'g')  # perfect result
        plt.legend([p1, p2, p3],
                   ["noisy measurement", "KF output", "ideal"], loc=4)
        plt.show()

def test_noisy_11d():
    f = kf.KalmanFilter(dim_x=2, dim_z=1)

    f.x = torch.tensor([2., 0])      # initial state (location and velocity)

    f.F = torch.tensor([[1., 1.],
                    [0., 1.]])    # state transition matrix

    f.H = torch.tensor([[1., 0.]])    # Measurement function
    f.P *= 1000.                  # covariance matrix
    f.R = torch.tensor([[5.]])                       # state uncertainty
    f.Q = torch.tensor([0.0001])                  # process uncertainty

    measurements = []
    results = []

    zs = []
    for t in range(100):
        # create measurement = t plus white noise
        z = t + torch.randn(1)*20
        zs.append(z)

        # perform kalman filtering
        f.update(z)
        f.predict()

        # save data
        results.append(f.x[0])
        measurements.append(z)

    # plot data
    if DO_PLOT:
        p1, = plt.plot(measurements, 'r', alpha=0.5)
        p2, = plt.plot(results, 'b')
        p3, = plt.plot([0, 100], [0, 100], 'g')  # perfect result
        plt.legend([p1, p2, p3],
                   ["noisy measurement", "KF output", "ideal"], loc=4)

        plt.show()

if __name__ == "__main__":
    DO_PLOT = True
    test_noisy_1d()
    test_noisy_11d()