import modern_high_performance as mr
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

class tm:
    """
    Class to represent and manipulate 3d transformations easily.
    Utilizes Euler Angles and 4x4 Transformation Matrices
    """
    def __init__(self, initArr = np.eye((4))):
        """
        Initializes a new Transformation object
        If no initArr is supplied, an identity transform is returned
        initArr can be of the following types, with different behaviors
            1) tm: a copy of the tm object is returned
            2) len(3) list: a tm object representing an XYZ rotation is returned
            3) len(6) list: a tm object representing an XYZ translation and XYZ rotation is returned
            4) 4x4 transformation matrix: a tm object representing the given transform is returned
        Args:
            initArr: Optional - data to generate new transformation
        """
        if hasattr(initArr, 'TM'):
            #Returns a copy of the tm object
            self.TM = initArr.TM.copy()
            self.TAA = initArr.TAA.copy()
            return
        elif isinstance(initArr, list):
            #Generates tm from list
            if len(initArr) == 3:
                self.TAA = np.array([0, 0, 0, initArr[0], initArr[1], initArr[2]])
            else:
                self.TAA = np.array([initArr[0], initArr[1], initArr[2], initArr[3], initArr[4], initArr[5]])
            self.TAAtoTM()
            return
        else:
            if len(initArr) == 6:
                #Generates tm from numpy array
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
        """
        Spawns a new TM object. Useful when tm is not speifically imported
        Args:
            init: initialization argument
        Returns:
            tm: tm object
        """
        return tm(init)

    def TransformSqueezedCopy(self, TM):
        """
        In cases of dimension troubles, squeeze out the extra dimension
        Args:
            TM: input 4x4 matrix
        Returns:
            TM: return 4x4 matrix
        """
        self.TM = np.eye((4))
        for i in range(4):
            for j in range(4):
                self.TM[i,j] = TM[i,j]
        return self.TM

    def getQuat(self):
        """
        Returns quaternion
        Returns:
            quaternion from euler
        """
        return R.from_euler('xyz', self.TAA[3:6].reshape((3)))

    def setQuat(self, quat):
        """
        Sets TAA from quaternion
        Args:
            quat: input quaternion
        """
        self.TAA[3:6] = R.from_quat(quat).as_euler('xyz')
        self.TAAtoTM()

    def update(self):
        """
        Updates the tm object
        """
        if not np.all(self.TM == self._transform_matrix_old):
            self.TMtoTAA()
            self._transform_matrix_old = np.copy(self.TM)
        elif not np.all(self.TAA == self._transform_taa_old):
            self.TAAtoTM()
            self._transform_taa_old = np.copy(self.TAA)

    def AngleMod(self):
        """
        Truncates excessive rotations
        """
        refresh = 0
        for i in range(3,6):
            if abs(self.TAA[i,0]) > 2 * np.pi:
                refresh = 1
                self.TAA[i,0] = self.TAA[i,0] % (np.pi)
        if refresh == 1:
            self.TAAtoTM()

    def LeftHanded(self):
        """
        Converts to left handed representation, for interactions
        with programs that use this for whatever reason
        Returns:
            tm: itself, but left handed
        """
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

    def TripleUnit(self, lv=1):
        """
        Returns XYZ unit vectors based on current position
        Args:
            lv: length to magnify by
        Returns:
            vec1: vector X
            vec2: vector Y
            vec3: vector Z
        """
        xvec = np.zeros(6)
        yvec = np.zeros(6)
        zvec = np.zeros(6)

        xvec[0:3] = self.TM[0:3,0].flatten()
        yvec[0:3] = self.TM[0:3,1].flatten()
        zvec[0:3] = self.TM[0:3,2].flatten()

        vec1 = self + tm(xvec*lv)
        vec2 = self + tm(yvec*lv)
        vec3 = self + tm(zvec*lv)

        return vec1, vec2, vec3

    def LeftHandedQuat(self):
        """
        Returns Quaternion, but left handed
        """
        tmn = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]])
        #print(tmn)
        #t2= qy * qz * qx
        mat =(tm(tmn) @ self)
        t1 = Quaternion(matrix=self.gTM())

        #print(t2)

        return t1

    #Murray Quaternion Definitions
    def getTheta(self):
        """
        Get Trace thetas
        Returns:
            thetas
        """
        trace_r = mr.SafeTrace(self.TM[0:3,0:3])
        theta = np.arccos((trace_r- 1)/2)
        return theta

    def getOmega(self):
        """
        gets Omega
        Returns:
            Omega
        """
        return (1/(2*np.sin(self.getTheta())) *
            np.array([self.TM[2,1]-self.TM[1,2], self.TM[0,2]
            -self.TM[2,0], self.TM[1,0] -self.TM[0,1]]))

    def getQuaternion(self):
        """
        getQuaterion according to murray definitions
        Returns:
            Quaternion Representation
        """
        #Q = (cos(theta/2), wsin(\theta/2))
        sec = self.getOmega()*np.sin(self.getTheta()/2)
        Q = np.array([np.cos(self.getTheta()/2), sec[0], sec[1], sec[2]])
        return Q

    def QuatToR(self, quat):
        """
        Converts a quaternion to a euler angle tm representation
        Args:
            qaut: Quaternion
        Returns:
            tm object
        """
        theta = 2 * np.arccos(quat[0])
        if theta != 0:
            w = quat[1:]/np.sin(theta/2)
        else:
            w = np.zeros((3))
        new = tm([w[0], w[1], w[2]])
        print(new)


    def TAAtoTM(self):
        """
        Converts the TAA representation to TM representation to update the object
        """
        self.TAA = self.TAA.reshape((6))
        mres = mr.MatrixExp3(mr.VecToso3(self.TAA[3:6]))
        #return mr.RpToTrans(mres,self.TAA[0:3])
        self.TAA = self.TAA.reshape((6,1))
        self.TM = np.vstack((np.hstack((mres,self.TAA[0:3])),np.array([0,0,0,1])))
        #self.AngleMod()
        #print(tm)

    def TMtoTAA(self):
        """
        Converts the TM representation to TAA representation and updates the object
        """
        tm, trans =  mr.TransToRp(self.TM)
        ta = mr.so3ToVec(mr.MatrixLog3(tm))
        self.TAA = np.vstack((trans.reshape((3,1)), (ta.reshape((3,1)))))
    #Modern Robotics Ports
    def Adjoint(self):
        """
        Returns the adjoint representation of the 4x4 transformation matrix
        Returns:
            Adjoint
        """
        return mr.Adjoint(self.TM)

    def Exp6(self):
        """
        Returns the matrix exponential of the TAA
        Returns:
            Matrix exponential
        """
        return mr.MatrixExp6(mr.VecTose3(self.TAA))

    def gRot(self):
        """
        Returns the rotation matrix component of the 4x4 transformation matrix
        Returns:
            3x3 rotation matrix
        """
        return self.TM[0:3,0:3].copy()

    def gTAA(self):
        """
        Returns the TAA representation of the tm object
        Returns:
            TAA
        """
        return np.copy(self.TAA)

    def gTM(self):
        """
        Returns the 4x4 transformation matrix representation of the tm object
        Returns:
            4x4 tm
        """
        return np.copy(self.TM)

    def gPos(self):
        """
        Returns the len(3) XYZ position of the transforamtion
        Returns:
            position
        """
        return self.TAA[0:3]

    def sTM(self, TM):
        """
        Sets the 4x4 transformation matrix and updates the object
        Args:
            TM: 4x4 transformation matrix to be set
        """
        self.TM = TM
        self.TMtoTAA()

    def sTAA(self, TAA):
        """
        Sets the TAA version of the object and updates
        Args:
            TAA: New TAA to be set
        """
        self.TAA = TAA
        self.TAAtoTM()
        #self.AngleMod()
    #Regular Transpose
    def T(self):
        """
        Returns the transpose version of the tm
        Returns:
            tm.T
        """
        TM = self.TM.T
        return tm(TM)

    #Conjugate Transpose
    def cT(self):
        """
        Returns the conjugate transpose if transpose doesn't work, use this.
        Returns:
            TM.conj().T
        """
        TM = self.TM.conj().T
        return tm(TM)

    def inv(self):
        """
        Returns the inverse of the transform
        Returns:
            tm^-1
        """
        #Regular Inverse
        TM = mr.TransInv(self.TM)
        return tm(TM)

    def pinv(self):
        """
        Returns the pseudoinverse of the transform
        Returns:
            psudo inverse
        """
        #Psuedo Inverse
        TM = np.linalg.pinv(self.TM)
        return tm(TM)

    def copy(self):
        """
        Returns a deep copy of the object
        Returns:
            copy
        """
        copy = tm()
        copy.TM = np.copy(self.TM)
        copy.TAA = np.copy(self.TAA)
        return copy

    def set(self, ind, val):
        """
        Sets a specific index of the TAA to a value and then updates
        Args:
            ind: index to set
            val: val to set
        Returns:
            new version of self reference
        """
        self.TAA[ind] = val
        self.TAAtoTM()
        return self

    def approx(self):
        """
        Rounds self to 10 decimal places
        Returns:
            rounded TAA representation
        """
        return np.around(self.TAA, 10)

    #FLOOR DIVIDE IS OVERRIDDEN TO PERFORM MATRIX RIGHT DIVISION

    #OVERLOADED FUNCTIONS
    def __getitem__(self, ind):
        """
        Get an indexed slice of the TAA representation
        Args:
            ind: slice
        Returns:
            Slice
        """
        if isinstance(ind, slice):
            return self.TAA[ind]
        else:
            return self.TAA[ind,0]

    def __setitem__(self, ind, val):
        """
        Sets an indexed slice of the TAA representation and updates
        Args:
            ind: slice
            val: value(s)
        """
        if isinstance(val, np.ndarray) and val.shape == ((3,1)):
            self.TAA[ind] = val
        else:
            self.TAA[ind,0] = val
        self.TAAtoTM()

    def __floordiv__(self, a):
        """
        PERFORMS RIGHT MATRIX DIVISION IF A IS A MATRIX
        Otherwise performs elementwise floor division
        Args:
            a: object to divide by
        Returns:
            floor (or matrix right division) result
        """
        if isinstance(a, tm):
            return tm(np.linalg.lstsq(b.T(), self.T())[0].T)
        elif isinstance(a, numpy.ndarray):
            return tm(np.linalg.lstsq(a.T, self.T())[0].T)
        else:
            return tm(self.TAA // a)

    def __abs__(self):
        """
        Returns absolute valued variant of transform
        Returns:
            |TAA|
        """
        return tm(abs(self.TAA))

    def __sum__(self):
        """
        Returns sum of the values of the TAA representation
        Returns:
            sum(TAA)
        """
        return sum(self.TAA)

    def __add__(self, a):
        """
        Adds a value to a TM
        If it is another TM, adds the two TAA representations together and updates
        If it is an array of 6 values, adds to the TAA and updates
        If it is a scalar, adds the scalar to each element of the TAA and updates
        Args:
            a: other value
        Returns:
            this + a
        """
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
        """
        Adds a value to a TM
        If it is another TM, subs a from TAA and updates
        If it is an array of 6 values, subs a from TAA and updates
        If it is a scalar, subs the scalar from each element of the TAA and updates
        Args:
            a: other value
        Returns:
            this - a
        """
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
        """
        Performs matrix multiplication on 4x4 TAA objects
        Accepts either another tm, or a matrix
        Args:
            a: thing to multiply by
        Returns:
            TM * a
        """
        if isinstance(a, tm):
            return tm(self.TM @ a.TM)
        else:
            if isinstance(a, np.ndarray):
                return tm(self.TM @ a)

    def __rmatmul__(self, a):
        """
        Performs right matrix multiplication on 4x4 TAA objects
        Accepts either another tm, or a matrix
        Args:
            a: thing to multiply by
        Returns:
            a * TM
        """
        if isinstance(a, tm):
            return tm(a.TM @ self.TM)
        else:
            if isinstance(a, np.ndarray):
                return tm(a @ self.TM)
            return tm(a * self.TAA)

    def __mul__(self, a):
        """
        Multiplies by a matrix or scalar
        Args:
            a: other value
        Returns:
            TM * a
        """
        if isinstance(a, tm):
            return tm(self.TM @ a.TM)
        else:
            if isinstance(a, np.ndarray):
                return tm(self.TM * a)
            return tm(self.TAA * a)

    def __rmul__(self, a):
        """
        Performs right multiplication by a matrix or scalar
        Args:
            a: thing to multiply by
        Returns:
            a * TM
        """
        if isinstance(a, tm):
            return tm(a.TM @ self.TM)
        else:
            if isinstance(a, np.ndarray):
                return tm(a * self.TM)
            return tm(a * self.TAA)

    def __truediv__(self, a):
        """
        Performs elementwise division across a TAA object
        Args:
            a: value to divide by
        Returns
            tm: new tm with requested division
        """
        #Divide Elementwise from TAA
        return tm(self.TAA / a)

    def __eq__(self, a):
        """
        Checks for equality with another tm
        Args:
            a: Another object
        Returns:
            boolean
        """
        if ~isinstance(a, tm):
            return False
        if np.all(self.TAA == a.TAA):
            return True
        return False

    def __gt__(self, a):
        """
        Checks for greater than another tm
        Args:
            a: Another object
        Returns:
            boolean
        """
        if isinstance(a, tm):
            if np.all(self.TAA > a.TAA):
                return True
        else:
            if np.all(self.TAA > a):
                return True
        return False

    def __lt__(self, a):
        """
        Checks for less than another tm
        Args:
            a: Another object
        Returns:
            boolean
        """
        if isinstance(a, tm):
            if np.all(self.TAA < a.TAA):
                return True
        else:
            if np.all(self.TAA < a):
                return True
        return False

    def __le__(self, a):
        """
        Checks for less than or equa to another tm
        Args:
            a: Another object
        Returns:
            boolean
        """
        if self.__lt__(a) or self.__eq__(a):
            return True
        return False

    def __ge__(self, a):
        """
        Checks for greater than or equal to another tm
        Args:
            a: Another object
        Returns:
            boolean
        """
        if self.__gt__(a) or self.__eq__(a):
            return True
        return False

    def __ne__(self, a):
        """
        Checks for not equality with another tm
        Args:
            a: Another object
        Returns:
            boolean
        """
        return not self.__eq__(a)

    def __str__(self,dlen=6):
        """
        Creates a string from a tm object
        Returns:
            String: representation of transform
        """
        fst = '%.' + str(dlen) + 'f'
        return ("[ " + fst % (self.TAA[0,0]) + ", "+ fst % (self.TAA[1,0]) +
         ", "+ fst % (self.TAA[2,0]) + ", "+ fst % (self.TAA[3,0]) +
         ", "+ fst % (self.TAA[4,0]) + ", "+ fst % (self.TAA[5,0])+ " ]")
