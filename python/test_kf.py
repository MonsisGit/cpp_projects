from parameterized import parameterized_class

import unittest
import numpy as np

import kf as kf_python


@parameterized_class([
    { "KF": kf_python.KF }
])
class TestKF(unittest.TestCase):

    def test_kalman_initialization(self):
        kf = self.KF(0,1,0.1)
        kf.get_cov()
        kf.get_state()

    def test_shapes_state_covariance(self):
        kf = self.KF(1,1,0.1)
        kf.predict(1)
        self.assertEqual(kf.get_cov().shape,(2,2))
        self.assertEqual(kf.get_state().shape,(2,))
        

    def test_cov_increases_with_only_predictions(self):
        kf = self.KF(3,1,0.1)

        for i in range(10):
            det_before = np.linalg.det(kf.get_cov())
            kf.predict(1)
            det_after = np.linalg.det(kf.get_cov())
            self.assertGreater(det_after,det_before)

    def test_cov_decreases_with_only_update(self):
        kf = self.KF(3,1,0.1)
        kf.predict(1)
        cov_before_update = np.linalg.det(kf.get_cov())
        kf.update(1, 0.1)
        self.assertGreater(cov_before_update, np.linalg.det(kf.get_cov()))

    def test_track_history(self):
        kf = self.KF(3,1,0.1)
        kf.track_history()
        for i in range(10):
            kf.predict(1)
            kf.update(i, 0.1)
        x,p = kf.get_history()

if __name__ == '__main__':
    unittest.main()