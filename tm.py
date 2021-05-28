import modern_high_performance as mr
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

class tm:

    TM = 0
    TAA = 0
    Quat = 0

    def __init__(self, initArr = np.eye((4))):
        if hasattr(initArr, 'TM'):
            self.TM = initArr.TM.copy()
            self.TAA = initArr.TAA.copy()
            return
        elif isinstance(initArr, list):
            if len(initArr) == 3:
                self.TAA = np.array([0, 0, 0, initArr[0], initArr[1], initArr[2]])
            else:
                self.TAA = np.array([initArr[0], initArr[1], initArr[2], initArr[3], initArr[4], initArr[5]])
            self.TAAtoTM()
            return
        else:
            if len(initArr) == 6:
                self.TAA = initArr.reshape((6,1)).copy()
                self.TAAtoTM()
                return
            elif (len(initArr) == 1):
                if isinstance(initArr, np.ndarray):
                    if isinstance(initArr[0], tm):
                        self.TM = initArr[0].TM.copy()
                        self.TMtoTAA()
                        return
            else:
                self.TransformSqueezedCopy(initArr)
                self.TMtoTAA()
                return

    def spawnNew(self, init):
        return tm(init)

    def TransformSqueezedCopy(self, TM):
        self.TM = np.eye((4))
        for i in range(4):
            for j in range(4):
                self.TM[i,j] = TM[i,j]
        return self.TM

    def getQuat(self):
        return R.from_euler('xyz', self.TAA[3:6].reshape((3)))

    def setQuat(self, quat):
        self.TAA[3:6] = R.from_quat(quat).as_euler('xyz')
        self.TAAtoTM()

    def update(self):
        if not np.all(self.TM == self._transform_matrix_old):
            self.TMtoTAA()
            self._transform_matrix_old = np.copy(self.TM)
        elif not np.all(self.TAA == self._transform_taa_old):
            self.TAAtoTM()
            self._transform_taa_old = np.copy(self.TAA)

    def AngleMod(self):
        refresh = 0
        for i in range(3,6):
            if abs(self.TAA[i,0]) > 2 * np.pi:
                refresh = 1
                self.TAA[i,0] = self.TAA[i,0] % (np.pi)
        if refresh == 1:
            self.TAAtoTM()

    def LeftHanded(self):
        #t1 = Quaternion(matrix=self.TM)
        #t2 = Quaternion(-1 * t1[1], 1 * t1[0], -1 * t1[2], t1[3])
        #t3 = tm(t2.transformation_matrix)
        #print(t3)
        #qx = Quaternion([1, 0, 0, -self[3]])
        #qz = Quaternion([0, 0, 1, -self[4]])
        #qy = Quaternion([0, 1, 0, -self[5]])
        #print(t2)

        #t3 = tm(t2.transformation_matrix)
        #tm, trans =  mr.TransToRp(self.TM)
        #ta = mr.so3ToVec(mr.MatrixLog3(tm.T))
        #t3 = tm([ta[0], ta[1], ta[2]])
        tmn = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]])
        #print(tmn)
        #t2= qy * qz * qx
        t3 = tm(tmn) @ self

        x = np.arctan2(t3.TM[1,2], t3.TM[2,2])
        #y = np.arctan2(-self.TM[0,2], np.sqrt(self.TM[1,2]**2 + self.TM[2,2]**2))
        y = np.arcsin(t3.TM[2,0])
        z = np.arctan2(t3.TM[0,1], t3.TM[0,0])
        #z = np.arctan(t3.TM[0,1]/t3.TM[0,0])
        t3 = tm([0,0,0,0,0,0])
        t3[0] = self.TAA[0]
        t3[1] = - self.TAA[1]
        t3[2] = self.TAA[2]
        t3[3] = x
        t3[4] = y
        t3[5] = z



        return self

    def TripleUnit(self):
        tmn = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]])
        #print(tmn)
        #t2= qy * qz * qx
        t3 = tm(tmn) @ self
        xax = t3 @ tm([1, 0, 0, 0, 0, 0]) #Best so far -1, 1, -1
        yax = t3 @ tm([0, -1, 0, 0, 0, 0])
        zax = t3 @ tm([0, 0, 1, 0, 0, 0])
        vec1 = mr.Normalize((-t3[0:3] + xax[0:3]).reshape((3)))
        vec2 = mr.Normalize((-t3[0:3] + yax[0:3]).reshape((3)))
        vec3 = mr.Normalize((-t3[0:3] + zax[0:3]).reshape((3)))

        t3 = tm(tmn) @ self
        xax = self @ tm([1, 0, 0, 0, 0, 0]) #Best so far -1, 1, -1
        yax = self @ tm([0, -1, 0, 0, 0, 0])
        zax = self @ tm([0, 0, 1, 0, 0, 0])
        #xax = tm(tmn) @ xax
        #yax = tm(tmn) @ yax
        #yax = tm(tmn) @ zax
        #vec1 = (-self[0:3] + xax[0:3]).reshape((3))
        #vec2 = (-self[0:3] + yax[0:3]).reshape((3))
        #vec3 = (-self[0:3] + zax[0:3]).reshape((3))


        vec1[1] = - vec1[1]
        vec2[1] = - vec2[1]
        vec3[1] = - vec3[1]

        return vec1, vec2, vec3

    def LeftHandedQuat(self):
        tmn = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]])
        #print(tmn)
        #t2= qy * qz * qx
        mat =(tm(tmn) @ self)
        t1 = Quaternion(matrix=self.gTM())

        #print(t2)

        return t1

    #Murray Quaternion Definitions
    def getTheta(self):
        trace_r = mr.SafeTrace(self.TM[0:3,0:3])
        theta = np.arccos((trace_r- 1)/2)
        return theta

    def getOmega(self):
        return 1/(2*np.sin(self.getTheta())) * np.array([self.TM[2,1]-self.TM[1,2], self.TM[0,2]-self.TM[2,0], self.TM[1,0] -self.TM[0,1]])
    def getQuaternion(self):
        #Q = (cos(theta/2), wsin(\theta/2))
        sec = self.getOmega()*np.sin(self.getTheta()/2)
        Q = np.array([np.cos(self.getTheta()/2), sec[0], sec[1], sec[2]])
        return Q

    def QuatToR(self, quat):
        theta = 2 * np.arccos(quat[0])
        if theta != 0:
            w = quat[1:]/np.sin(theta/2)
        else:
            w = np.zeros((3))
        new = tm([w[0], w[1], w[2]])
        print(new)


    def TAAtoTM(self):
        self.TAA = self.TAA.reshape((6))
        mres = mr.MatrixExp3(mr.VecToso3(self.TAA[3:6]))
        #return mr.RpToTrans(mres,self.TAA[0:3])
        self.TAA = self.TAA.reshape((6,1))
        self.TM = np.vstack((np.hstack((mres,self.TAA[0:3])),np.array([0,0,0,1])))
        #self.AngleMod()
        #print(tm)

    def TMtoTAA(self):
        tm, trans =  mr.TransToRp(self.TM)
        ta = mr.so3ToVec(mr.MatrixLog3(tm))
        self.TAA = np.vstack((trans.reshape((3,1)), (ta.reshape((3,1)))))
    #Modern Robotics Ports
    def Adjoint(self):
        return mr.Adjoint(self.TM)

    def Exp6(self):
        return mr.MatrixExp6(mr.VecTose3(self.TAA))

    def gRot(self):
        return self.TM[0:3,0:3].copy()

    def gTAA(self):
        return np.copy(self.TAA)

    def gTM(self):
        return np.copy(self.TM)

    def gPos(self):
        return self.TAA[0:3]

    def sTM(self, TM):
        self.TM = TM
        self.TMtoTAA()

    def sTAA(self, TAA):
        self.TAA = TAA
        self.TAAtoTM()
        #self.AngleMod()
    #Regular Transpose
    def T(self):
        TM = self.TM.T
        return tm(TM)

    #Conjugate Transpose
    def cT(self):
        TM = self.TM.conj().T
        return tm(TM)

    def inv(self):
        #Regular Inverse
        TM = mr.TransInv(self.TM)
        return tm(TM)

    def pinv(self):
        #Psuedo Inverse
        TM = np.linalg.pinv(self.TM)
        return tm(TM)

    def copy(self):
        copy = tm()
        copy.TM = np.copy(self.TM)
        copy.TAA = np.copy(self.TAA)
        return copy

    def set(self, ind, val):
        self.TAA[ind] = val
        self.TAAtoTM()
        return self

    def approx(self):
        return np.around(self.TAA, 10)

    #FLOOR DIVIDE IS OVERRIDDEN TO PERFORM MATRIX RIGHT DIVISION

    #OVERLOADED FUNCTIONS
    def __getitem__(self, ind):
        if isinstance(ind, slice):
            return self.TAA[ind]
        else:
            return self.TAA[ind,0]

    def __setitem__(self, ind, val):
        if isinstance(val, np.ndarray) and val.shape == ((3,1)):
            self.TAA[ind] = val
        else:
            self.TAA[ind,0] = val
        self.TAAtoTM()

    def __floordiv__(self, a):
        if isinstance(a, tm):
            return tm(np.linalg.lstsq(b.T(), self.T())[0].T)
        elif isinstance(a, numpy.ndarray):
            return tm(np.linalg.lstsq(a.T, self.T())[0].T)
        else:
            return tm(self.TAA // a)

    def __abs__(self):
        return tm(abs(self.TAA))

    def __sum__(self):
        return sum(self.TAA)

    def __add__(self, a):
        if isinstance(a, tm):
            return tm(self.TAA + a.TAA)
        else:
            if isinstance(a, np.ndarray):
                if len(a) == 6:
                    return tm(self.TAA + a)
                else:
                    return self.TAA + a
            else:
                return self.TAA + a

    def __sub__(self, a):
        if isinstance(a, tm):
            return tm(self.TAA - a.TAA)
        else:
            if isinstance(a, np.ndarray):
                if len(a) == 6:
                    return tm(self.TAA - a.reshape((6,1)))
                else:
                    return self.TAA - a
            else:
                return self.TAA - a

    def __matmul__(self, a):
        if isinstance(a, tm):
            return tm(self.TM @ a.TM)
        else:
            if isinstance(a, np.ndarray):
                return tm(self.TM @ a)

    def __rmatmul__(self, a):
        if isinstance(a, tm):
            return tm(a.TM @ self.TM)
        else:
            if isinstance(a, np.ndarray):
                return tm(a @ self.TM)
            return tm(a * self.TAA)

    def __mul__(self, a):
        if isinstance(a, tm):
            return tm(self.TM @ a.TM)
        else:
            if isinstance(a, np.ndarray):
                return tm(self.TM * a)
            return tm(self.TAA * a)

    def __rmul__(self, a):
        if isinstance(a, tm):
            return tm(a.TM @ self.TM)
        else:
            if isinstance(a, np.ndarray):
                return tm(a * self.TM)
            return tm(a * self.TAA)

    def __truediv__(self, a):
        #Divide Elementwise from TAA
        return tm(self.TAA / a)

    def __eq__(self, a):
        if ~isinstance(a, tm):
            return False
        if np.all(self.TAA == a.TAA):
            return True
        return False

    def __gt__(self, a):
        if isinstance(a, tm):
            if np.all(self.TAA > a.TAA):
                return True
        else:
            if np.all(self.TAA > a):
                return True
        return False

    def __lt__(self, a):
        if isinstance(a, tm):
            if np.all(self.TAA < a.TAA):
                return True
        else:
            if np.all(self.TAA < a):
                return True
        return False

    def __le__(self, a):
        if self.__lt__(a) or self.__eq__(a):
            return True
        return False

    def __ge__(self, a):
        if self.__gt__(a) or self.__eq__(a):
            return True
        return False

    def __ne__(self, a):
        return not self.__eq__(a)

    def __str__(self,dlen=6):
        fst = '%.' + str(dlen) + 'f'
        return ("[ " + fst % (self.TAA[0,0]) + ", "+ fst % (self.TAA[1,0]) +
         ", "+ fst % (self.TAA[2,0]) + ", "+ fst % (self.TAA[3,0]) +
         ", "+ fst % (self.TAA[4,0]) + ", "+ fst % (self.TAA[5,0])+ " ]")
