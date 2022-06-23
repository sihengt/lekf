import torch
import torch.nn as nn

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
    compute_log_likelihood : bool (default = True)
        Computes log likelihood by default, but this can be a slow
        computation, so if you never use it you can turn this computation
        off.
    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.
    P : numpy.array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.
    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convenience; they store the  prior and posterior of the
        current epoch. Read Only.
    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.
    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.
    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.
    z : numpy.array
        Last measurement used in update(). Read only.
    R : numpy.array(dim_z, dim_z)
        Measurement noise covariance matrix. Also known as the
        observation covariance.
    Q : numpy.array(dim_x, dim_x)
        Process noise covariance matrix. Also known as the transition
        covariance.
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
    log_likelihood : float
        log-likelihood of the last measurement. Read only.
    likelihood : float
        likelihood of last measurement. Read only.
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
    mahalanobis : float
        mahalanobis distance of the innovation. Read only.
    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv
        This is only used to invert self.S. If you know it is diagonal, you
        might choose to set it to filterpy.common.inv_diagonal, which is
        several times faster than numpy.linalg.inv for diagonal matrices.
    alpha : float
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
    References
    ----------
    .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
       p. 208-212. (2006)
    .. [2] Roger Labbe. "Kalman and Bayesian Filters in Python"
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

        # self._alpha_sq = 1.                     # (UNUSED) fading memory control
        # self.M = torch.zeros((dim_x, dim_z))    # (UNUSED) process-measurement cross correlation
        self.z = torch.zeros((self.dim_z,1)).T  # observations

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = torch.zeros((dim_x, dim_z)) # kalman gain
        self.y = torch.zeros((dim_z, 1))
        self.S = torch.zeros((dim_z, dim_z)) # system uncertainty
        self.SI = torch.zeros((dim_z, dim_z)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = torch.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.detach().clone()
        self.P_prior = self.P.detach().clone()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.detach().clone()
        self.P_post = self.P.detach().clone()

        self.inv = torch.linalg.inv

    def predict(self, u=None):
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
        # x = Fx + Bu
        if u:
            self.x = torch.matmul(self.F, self.x) + torch.matmul(self.B, u)
        else:
            self.x = torch.matmul(self.F, self.x)

        # P = FPF' + Q
        self.P = torch.matmul(torch.matmul(self.F, self.P), self.F.T) + self.Q

        # save prior
        self.x_prior = self.x.detach().clone()
        self.P_prior = self.P.detach().clone()

    def update(self, z):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
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

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - torch.matmul(self.H, self.x)

        # common subexpression for speed
        PHT = torch.matmul(self.P, self.H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = torch.matmul(self.H, PHT) + self.R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = torch.matmul(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + torch.matmul(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - torch.matmul(self.K, self.H)
        self.P = torch.matmul(torch.matmul(I_KH, self.P), I_KH.T) + torch.matmul(torch.matmul(self.K, self.R), self.K.T)

        # save measurement and posterior state
        self.z = z.detach().clone()
        self.x_post = self.x.detach().clone()
        self.P_post = self.P.detach().clone()

    def forward(self, state):
        # call both predict and update here. 
        pass

if __name__ == "__main__":
    kf = KalmanFilter(3,3,3)