import numpy as np
from typing import Tuple

class KF():
    def __init__(self, initial_pos: float,
                initial_vel: float,
                accel_variance: float
                ) -> None:
    
        self._x = np.array([initial_pos, initial_vel])
        self._p = np.eye(2)
        self._accel_variance = accel_variance
        self.time_steps = 0
        self._x_history = []
        self._p_history = []
        self._track_history = False

    def get_state(self) -> np.array:
        return self._x
    
    def get_cov(self) -> np.array:
        return self._p

    def track_history(self) -> None:
        self._track_history = True

    def get_history(self) -> Tuple[list, list]:
        return self._x_history, self._p_history

    def _update_state_covariance(self, new_x: np.array, new_p: np.array, is_update: bool = False) -> None:

        self._x = new_x
        self._p = new_p

        if self._track_history and not is_update:
            self._x_history.append(list(new_x))
            self._p_history.append(list(new_p))

    def predict(self, dt: float) -> None:

        F = np.array([[1,dt],[0,1]]) #state transition matrix
        G = np.array([0.5*dt**2, dt]).reshape(2,1)

        estimate_x = F.dot(self._x) #no B*u, since no control input
        estimate_p = F.dot(self._p).dot(F.T) + G.dot(G.T)*self._accel_variance

        self._update_state_covariance(estimate_x, estimate_p)

        self.time_steps += dt

    def update(self, measurement: float,
                     measurement_variance: float):

        z = np.array([measurement])
        R = np.array([measurement_variance])

        H = np.array([1,0]).reshape(1,2)
        y = z - H.dot(self._x)

        S = H.dot(self._p).dot(H.T) + R
        K = self._p.dot(H.T) * np.linalg.inv(S)

        new_x = self._x + (K*y)[:,0]
        new_p = (np.eye(2)-K.dot(H)).dot(self._p)

        self._update_state_covariance(new_x, new_p, True)


