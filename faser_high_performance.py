import numpy as np
import numba
from numba import jit
from modern_high_performance import *

#Parallel Disabled For Now. Something is wrong with my installation.
@jit(nopython=True, cache=True)#, parallel=True)
def IKinSpaceConstrained(Slist, M, T, thetalist, eomg, ev, jointMins, jointMaxs, maxiterations):
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = np.dot(Adjoint(Tsb), se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
    #print(mhp.MatrixLog6(np.dot(mhp.TransInv(Tsb), T)), "Test")
    err = np.linalg.norm(Vs[0:3]) > eomg or np.linalg.norm(Vs[3:6]) > ev
    if np.isnan(Vs).any():
        err = True
    i = 0
    while err and i < maxiterations:
        jspace = JacobianSpace(Slist, thetalist)
        invj = np.linalg.pinv(jspace)
        newtheta =np.dot(invj, Vs)
        thetalist = thetalist + newtheta
        for j in range(len(thetalist)):
            if thetalist[j] < jointMins[j]:
                thetalist[j] = jointMins[j]
            if thetalist[j] > jointMaxs[j]:
                thetalist[j] = jointMaxs[j];
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = np.dot(Adjoint(Tsb), se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
        err = np.linalg.norm(Vs[0:3]) > eomg or np.linalg.norm(Vs[3:6]) > ev
        if np.isnan(Vs).any():
            err = True
    success = not err

    return thetalist, success

@jit(nopython=True, cache=True, parallel=True)
def SPIKinSpace(bottomT, topT, bottomJoints, topJoints, bJS, tJS):
    lengths = np.zeros((6, 1))

    #Perform Inverse Kinematics
    for i in range(6):
        bJS[0:3, i] = TrVec(bottomT, bottomJoints[0:3, i])
        tJS[0:3, i] = TrVec(topT, topJoints[0:3, i])
        t_len = Norm(tJS[0:3, i]-bJS[0:3, i])
        lengths[i] = t_len

    return lengths, bJS, tJS

#Majority of following section adapted from work by Jak-O-Shadows, Under MIT License
@jit(nopython=True, cache=True)
def SPFKinSpaceR(bottomT, L, h, bPos, pPos, maxIters, tol_f, tol_a, lmin):
    iterNum = 0
    #a = np.zeros((6))
    #a[2] = h
    a = h
    while iterNum < maxIters:
            iterNum += 1
            angs = np.zeros((6))
            j = 0
            a[3:6] = AngleMod(a[3:6])
            for i in range(3, 6):
                angs[j] = (np.cos(a[i]))
                angs[j+1] = (np.sin(a[i]))
                j+=2

            t = a[2]
            if t < lmin/2:
                a[2] = lmin/2.0
            #angs[5] = np.clip(angs[5], -np.pi/3.85, np.pi/3.85)
            #Must translate platform coordinates into base coordinate system
            #Calculate rotation matrix elements
            Rzyx = (MatrixExp3(VecToso3(a[3:6])))

            #disp(Rzyx, "Rzyx")
            #Hence platform sensor points with respect to the base coordinate system
            xbar = a[0:3] - bPos

            #Hence orientation of platform wrt base
            uvw = np.zeros(pPos.shape)
            for i in range(6):
                uvw[i, :] = np.dot(Rzyx, pPos[i, :])


            l_i = np.sum(np.square(xbar + uvw), 1)
            #Hence find value of objective function
            #The calculated lengths minus the actual length
            f = -1 * (l_i - np.square(L))
            sumF = np.sum(np.abs(f))
            if sumF < tol_f:
                #success!
                #print("Converged!")
                break

            #As using the newton-raphson matrix, need the jacobian (/hessian?) matrix
            #Using paper linked https://jak-o-shadows.github.io/electronics/stewart-gough/kinematics-stewart-gough-platform.pdf
            dfda = np.zeros((6, 6))
            dfda[:, 0:3] = 2*(xbar + uvw)
            for i in range(6):
                #dfda4 is swapped with dfda6 for magic reasons!
                dfda[i, 5] = 2*(-xbar[i, 0]*uvw[i, 1] + xbar[i, 1]*uvw[i, 0]) #dfda4
                dfda[i, 4] = 2*((-xbar[i, 0]*angs[4] + xbar[i, 1]*angs[5])*uvw[i, 2] \
                                - (pPos[i, 0]*angs[2] + pPos[i, 1]*angs[3]*angs[1])*xbar[i, 2]) #dfda5
                dfda[i, 3] = 2*pPos[i, 1]*(np.dot(xbar[i,:],Rzyx[:, 2])) #dfda
            #disp(dfda, "Dfda")
            #disp(np.linalg.inv(self.InverseJacobianSpace(self.gbottomT(), self.gtopT())))
            #Hence solve system for delta_{a} - The change in lengths
            delta_a = np.linalg.solve(dfda, f)

            if abs(np.sum(delta_a)) < tol_a:
                #print ("Small change in lengths -- converged?")
                break
            a = a + delta_a
    return a, iterNum

#Performs tv = TM*vec and removes the 1
@jit(nopython=True, cache=True)
def TrVec(TM, vec):
    b = np.ones((4))
    b[0:3] = vec
    trvh = TM @ b
    return trvh[0:3]
