import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:

    def __init__(self, mean_init, cov_init, command_var, measurement_var, Δt=1):
        self.mean = mean_init
        self.cov = cov_init
        self.A = np.array([[1, Δt], [0, 1]])
        self.B = np.array([0.5 * (Δt ** 2), Δt]).reshape(2, 1)
        self.C = np.array([1, 0]).reshape(1, 2)
        # noise covariance matrix
        self.R = self.B @ np.array(command_var).reshape(1, 1) @ self.B.T 
        self.Q = np.array([measurement_var]).reshape(1, 1)
        self.I = np.eye(self.A.shape[0])

    def plot_distributions(self) -> None:
       pass 

    def predict(self, command):
        self.mean = self.A @ self.mean + self.B @ command
        self.cov = self.A @ self.cov @ self.A.T + self.R 

    def correct(self, measurement):
        innovation = measurement - self.C @ self.mean
        # innovation covariance matrix
        S_inv = np.linalg.inv(self.C @ self.cov @ self.C.T + self.Q)
        kalman_gain = self.cov @ self.C.T  @ S_inv 
        self.mean = self.mean + kalman_gain @ innovation
        self.cov = (self.I - kalman_gain @ self.C) @ self.cov
    
    def update(self, command, measurement=None):
        self.predict(command)
        if measurement is not None:
            self.correct(measurement)
    


if __name__ == '__main__':
    kf = KalmanFilter()
    mp = np.array([2, 3.25]).reshape(2, 1)
    vp = np.array([[0.86, 0], [0, 4]])
    m_t, v_t = kf.update(mp, vp)
    print('m_t: ', m_t.shape)
    print(m_t)
    print('v_t: ', v_t.shape)
    print(v_t)

