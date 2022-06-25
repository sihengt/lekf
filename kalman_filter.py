# Adapted from Roger Labbe's FilterPy
# https://github.com/rlabbe/filterpy
import torch
import torch.nn as nn
import torch.optim as optim

class KalmanFilter(nn.Module):
    """
    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
        This is used to set the default size of P, Q, and u
    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.
    dim_u : int (optional)
        size of the control input, if it is being used.
        Default value of 0 indicates it is not used.
    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.
    P : numpy.array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.
    R : numpy.array(dim_z, dim_z)
        Measurement noise covariance matrix. Also known as the
        observation covariance.
    Q : numpy.array(dim_x, dim_x)
        Process noise covariance matrix. Also known as the transition
        covariance.    R : numpy.array(dim_z, dim_z)
        Measurement noise covariance matrix. Also known as the
        observation covariance.

    F : numpy.array()
        State Transition matrix. Also known as `A` in some formulation.
    H : numpy.array(dim_z, dim_x)
        Measurement function. Also known as the observation matrix, or as `C`.
    y : numpy.array
        Residual of the update step. Read only.
    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.
    S :  numpy.array
        System uncertainty (P projected to measurement space). Read only.
    SI :  numpy.array
        Inverse system uncertainty. Read only.
    References
    ----------
    .. [1] Roger Labbe. "Kalman and Bayesian Filters in Python"
       https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """
    def __init__(self, dim_x, dim_z, dim_u=0):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = torch.zeros((dim_x, 1))        # state
        self.P = torch.eye(dim_x)               # uncertainty covariance
        self.Q = torch.eye(dim_x)               # TODO: process uncertainty
        self.B = None                           # control transition matrix
        self.F = torch.eye(dim_x)               # state transition matrix
        self.H = torch.zeros((dim_z, dim_x))    # measurement function
        self.R = torch.eye(dim_z)               # measurement uncertainty

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = torch.zeros((dim_x, dim_z)) # kalman gain
        self.y = torch.zeros((dim_z, 1))
        self.S = torch.zeros((dim_z, dim_z)) # system uncertainty
        self.SI = torch.zeros((dim_z, dim_z)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = torch.eye(dim_x)

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array, default 0
            Optional control vector.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.09`vcA<5
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q

        # x = Fx + Bu
        if B and u:
            self.x = (F @ self.x) + (B @ u)   
        else:
            self.x = F @ self.x

        # P = FPF' + Q
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing is computed.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """
        if R is None:
            R = self.R
        if H is None:
            # Note: removed weird reshape function.
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - H @ self.x

        # common subexpression for speed
        PHT = self.P @ H.T

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = H @ PHT + R
        self.SI = torch.inverse(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = PHT @ self.SI

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + self.K @ self.y

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - self.K @ self.H
        self.P = ((I_KH @ self.P) @ I_KH.T) + ((self.K @ R) @ self.K.T)

    def forward(self, u, z, B=None, F=None, Q=None, R=None, H=None):
        # call both predict and update here.
        self.predict(u)
        self.update(z)
        return self.x

if __name__ == "__main__":
    pass