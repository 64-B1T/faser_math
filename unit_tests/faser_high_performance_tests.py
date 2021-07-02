import unittest
import sys
import numpy as np
sys.path.append('..')
sys.path.append('../..')
import faser_general as fsr
import faser_high_performance as fmr
import modern_robotics as mr

class TestFASERHighPerformance(unittest.TestCase):

    def testTransToRP(self):
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
        expected_r = np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]])
        expected_p = np.array([0, 0, 3])

        mr_r, mr_p = mr.TransToRp(T)

        self.assertTrue(np.allclose(mr_r, expected_r))
        self.assertTrue(np.allclose(mr_p, expected_p))

        fmr_r, fmr_p = fmr.TransToRp(T)

        self.assertTrue(np.allclose(fmr_r, expected_r))
        self.assertTrue(np.allclose(fmr_p, expected_p))

    def testso3ToVec(self):
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
        expected_out = np.array([1, 2, 3])

        self.assertTrue(np.allclose(mr.so3ToVec(so3mat), expected_out))
        self.assertTrue(np.allclose(fmr.so3ToVec(so3mat), expected_out))

    def testMatrixLog3(self):
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
        expected_out = np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
        self.assertTrue(np.allclose(mr.MatrixLog3(R), expected_out))
        self.assertTrue(np.allclose(fmr.MatrixLog3(R), expected_out))

        R = np.array([[0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(np.allclose(fmr.MatrixLog3(R), mr.MatrixLog3(R)))

        for i in range(100):
            R = np.random.uniform(-1.0, 1.0, (3,3))
            self.assertTrue(np.allclose(fmr.MatrixLog3(R), mr.MatrixLog3(R)))

if __name__ == '__main__':
    unittest.main()
