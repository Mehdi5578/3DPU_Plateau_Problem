import unittest

from _3dpu import *

class Test3DPU(unittest.TestCase):

    def test_potential_neighbors1(self):
        sr = SpinnedResidual(1, Residual(0, 1, np.array([2, 3, 2])))
        res = [
            SpinnedResidual(spin=1, res=Residual(ax=0, ori=1, pos=np.array([3, 3, 2]))),
            SpinnedResidual(spin=-1, res=Residual(ax=1, ori=-1, pos=np.array([2, 3, 2]))),
            SpinnedResidual(spin=1, res=Residual(ax=1, ori=1, pos=np.array([2, 4, 2]))),
            SpinnedResidual(spin=-1, res=Residual(ax=2, ori=-1, pos=np.array([2, 3, 2]))),
            SpinnedResidual(spin=1, res=Residual(ax=2, ori=1, pos=np.array([2, 3, 3])))
        ]
        self.assertEqual(potential_neighbors(sr), res)
    
    def test_potential_neighbors2(self):    
        sr = SpinnedResidual(1, Residual(0, 1, np.array([2, 3, 2])))
        res = [
            SpinnedResidual(spin=1, res=Residual(ax=0, ori=1, pos=np.array([1, 3, 2]))),
            SpinnedResidual(spin=-1, res=Residual(ax=1, ori=-1, pos=np.array([1, 4, 2]))),
            SpinnedResidual(spin=1, res=Residual(ax=1, ori=1, pos=np.array([1, 3, 2]))),
            SpinnedResidual(spin=-1, res=Residual(ax=2, ori=-1, pos=np.array([1, 3, 2]))),
            SpinnedResidual(spin=1, res=Residual(ax=2, ori=1, pos=np.array([1, 3, 4])))
        ]
        self.assertEqual(potential_neighbors(sr, reverse=True), res)
    
if __name__ == '__main__':
    unittest.main()