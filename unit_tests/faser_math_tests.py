import unittest
import sys
import numpy as np
sys.path.append('..')
sys.path.append('../..')
import faser_general as fsr
import faser_high_performance as fmr
import modern_robotics as mr
from faser_transform import tm

class TestFASERGeneral(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def testTAAtoTM(self):
        goal_tm = np.array([[1, 0, 0, 1],[0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        start_taa = np.array([1, 2, 3, 0, 0, 0])

        res_tm = fsr.TAAtoTM(start_taa)
        self.assertTrue(np.allclose(goal_tm, res_tm))
        start_taa = start_taa.T
        res_tm = fsr.TAAtoTM(start_taa)
        self.assertTrue(np.allclose(goal_tm, res_tm))

        goal_tm = np.array([[0, -1, 0, 1],[1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        start_taa = np.array([1, 2, 3, 0, 0, np.pi/2])

        res_tm = fsr.TAAtoTM(start_taa)
        #print(res_tm)
        self.assertTrue(np.allclose(goal_tm, res_tm))
        start_taa = start_taa.T
        res_tm = fsr.TAAtoTM(start_taa)
        self.assertTrue(np.allclose(goal_tm, res_tm))


    def testTMtoTAA(self):
        start_tm = np.array([[1, 0, 0, 1],[0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
        goal_taa = np.array([1, 2, 3, 0, 0, 0])

        res_taa = fsr.TMtoTAA(start_tm).flatten()
        self.assertTrue(np.allclose(res_taa, goal_taa, atol=.01))

        start_tm = np.array([[0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0]], dtype=float)
        goal_taa = np.array([1, 2, 3, 0, 0, np.pi/2])

        res_taa = fsr.TMtoTAA(start_tm).flatten()
        #print(goal_taa)
        #print(res_taa)
        self.assertTrue(np.allclose(res_taa, goal_taa, atol=.01))

        res_taa_2 = fsr.TMtoTAA(fsr.TAAtoTM(goal_taa)).flatten()
        self.assertTrue(np.allclose(res_taa, res_taa_2, atol=.01))
        #print(res_taa_2)


if __name__ == '__main__':
    unittest.main()
