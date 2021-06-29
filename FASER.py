#import whatever we need, make note of what packages to install
import math
from modern_robotics_numba import *
#import modern_high_performance as mr
import modern_robotics as mrs
import numpy as np
import scipy as sci
#import robopy as rp
import scipy.linalg as ling
from tm import tm

import copy


#TRANSFORMATION MATRIX MANIPULATIONS
def TAAtoTM(transaa):
    """Short summary.

    Args:
        transaa (type): Description of parameter `transaa`.

    Returns:
        type: Description of returned object.

    """
    """Short summary.

    Args:
        transaa (type): Description of parameter `transaa`.

    Returns:
        type: Description of returned object.

    """
    transaa = transaa.reshape((6))
    mres = mr.MatrixExp3(mr.VecToso3(transaa[3:6]))
    #return mr.RpToTrans(mres, transaa[0:3])
    transaa = transaa.reshape((6, 1))
    tm = np.vstack((np.hstack((mres, transaa[0:3])), np.array([0, 0, 0, 1])))
    #print(tm)
    return tm

def PlaneFrom3Tms(tm1, tm2, tm3):
    """
    Creates the equation of a plane from three points
    Args:
        tm1: tm or vector for point 1
        tm2: tm or vector for point 2
        tm3: tm or vector for point 3
    Returns:
        a, b, c, d: equation cooficients of a plane
    """
    p1 = np.array(tm1[0:3]).flatten()
    p2 = np.array(tm2[0:3]).flatten()
    p3 = np.array(tm3[0:3]).flatten()

    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane

    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return a, b, c, d

def PlaneTMSFromOne(tm1):
    """
    Create plane TM points from one Transform (using unit vectors)
    Args:
        tm1: transform to place plane on
    Returns:
        a, b, c: Plane basis points
    """
    a, b, x = tm1.TripleUnit()
    return tm1, b, c

def Mirror(origin, mirrorPoint):
    """
    Mirrors a point about a plane
    Args:
        origin: point at origin
        mirrorPoint: tm describing plane to mirror over
    Returns:
        mirrored Point
    """
    t1, t2, t3 = PlaneTMSFromOne(origin)
    a, b, c, d = PlaneFrom3Tms(t1, t2, t3)
    x1 = mirrorPoint[0]
    y1 = mirrorPoint[1]
    z1 = mirrorPoint[2]
    k =(-a * x1-b * y1-c * z1-d)/float((a * a + b * b + c * c))
    x2 = a * k + x1
    y2 = b * k + y1
    z2 = c * k + z1
    x3 = 2 * x2-x1
    y3 = 2 * y2-y1
    z3 = 2 * z2-z1
    return tm([x3, y3, z3, 0, 0, 0])

def TAAtoTMLegacy(transaa):
    """
    Transforms a TAA representation to a TM one
    Args:
        transaa: taa representation
    Returns:
        tm: TM representation
    """
    transaa = transaa.reshape((6))
    mres = mrs.MatrixExp3(mrs.VecToso3(transaa[3:6]))
    #return mr.RpToTrans(mres, transaa[0:3])
    transaa = transaa.reshape((6, 1))
    tm = np.vstack((np.hstack((mres, transaa[0:3])), np.array([0, 0, 0, 1])))
    #print(tm)
    return tm

def TMtoTAA(tm):
    """
    Converts a 4x4 transformation matrix to TAA representation
    Args:
        tm: tm to be convverted
    Returns:
        TAA representation
    """
    tm, trans =  mr.TransToRp(tm)
    ta = mr.so3ToVec(mr.MatrixLog3(tm))
    return np.vstack((trans.reshape((3, 1)), AngleMod(ta.reshape((3, 1)))))

def TMtoTAALegacy(tm):
    """Short summary.

    Args:
        tm (type): Description of parameter `tm`.

    Returns:
        type: Description of returned object.

    """
    tm, trans =  mrs.TransToRp(tm)
    ta = mrs.so3ToVec(mrs.MatrixLog3(tm))
    return np.vstack((trans.reshape((3, 1)), AngleMod(ta.reshape((3, 1)))))

def LocalToGlobal(reference, rel):
    """Short summary.

    Args:
        reference (type): Description of parameter `reference`.
        rel (type): Description of parameter `rel`.

    Returns:
        type: Description of returned object.

    """
    return tm(mr.LocalToGlobal(reference.gTAA(), rel.gTAA()))

def GlobalToLocal(reference, rel):
    """Short summary.

    Args:
        reference (type): Description of parameter `reference`.
        rel (type): Description of parameter `rel`.

    Returns:
        type: Description of returned object.

    """
    return tm(mr.GlobalToLocal(reference.gTAA(), rel.gTAA()))

def TrVec(TM, vec):
    """Short summary.

    Args:
        TM (type): Description of parameter `TM`.
        vec (type): Description of parameter `vec`.

    Returns:
        type: Description of returned object.

    """
    #Performs tv = TM*vec and removes the 1
    TM = TM.TM
    b = np.array([1.0])
    n = np.concatenate((vec, b))
    trvh = TM @ n
    return trvh[0:3]

def TMMidRotAdjust(inTF, tm1, tm2, mode = 0):
    """Short summary.

    Args:
        inTF (type): Description of parameter `inTF`.
        tm1 (type): Description of parameter `tm1`.
        tm2 (type): Description of parameter `tm2`.
        mode (type): Description of parameter `mode`.

    Returns:
        type: Description of returned object.

    """
    toAdj = inTF.copy()
    if mode != 1:
        t_mid = TMMidPoint(tm1, tm2)
        toAdj[3:6] = t_mid[3:6]
    else:
        toAdj[3:6] = RotFromVec(tm1, tm2)[3:6]
    return toAdj


def TMMidPointEx(taa1, taa2):
    """Short summary.

    Args:
        taa1 (type): Description of parameter `taa1`.
        taa2 (type): Description of parameter `taa2`.

    Returns:
        type: Description of returned object.

    """
    return (taa1 + taa2)/2

def TMMidPoint(taa1, taa2):
    """Short summary.

    Args:
        taa1 (type): Description of parameter `taa1`.
        taa2 (type): Description of parameter `taa2`.

    Returns:
        type: Description of returned object.

    """
    taar = np.zeros((6, 1))
    taar[0:3] = (taa1[0:3] + taa2[0:3])/2
    R1 = mr.MatrixExp3(mr.VecToso3(taa1[3:6].reshape((3))))
    R2 = mr.MatrixExp3(mr.VecToso3(taa2[3:6].reshape((3))))
    Re = (R1 @ (R2.conj().T)).conj().T
    Re2 = mr.MatrixExp3(mr.VecToso3(mr.so3ToVec(mr.MatrixLog3((Re)/2))))
    rmid = Re2 @ R1
    taar[3:6] = mr.so3ToVec(mr.MatrixLog3((rmid))).reshape((3, 1))
    return tm(taar)

def Error(tm1, tm2):
    """Short summary.

    Args:
        tm1 (type): Description of parameter `tm1`.
        tm2 (type): Description of parameter `tm2`.

    Returns:
        type: Description of returned object.

    """
    return abs(tm1 - tm2)

def GeometricError(tm1, tm2):
    """Short summary.

    Args:
        tm1 (type): Description of parameter `tm1`.
        tm2 (type): Description of parameter `tm2`.

    Returns:
        type: Description of returned object.

    """
    return GlobalToLocal(tm2, tm1)


def Distance(pos1, pos2):
    """Short summary.

    Args:
        pos1 (type): Description of parameter `pos1`.
        pos2 (type): Description of parameter `pos2`.

    Returns:
        type: Description of returned object.

    """
    try:
        d = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2)
    except:
        d = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    return d

def ArcDistance(pos1, pos2):
    """Short summary.

    Args:
        pos1 (type): Description of parameter `pos1`.
        pos2 (type): Description of parameter `pos2`.

    Returns:
        type: Description of returned object.

    """
    rpos = GlobalToLocal(pos1, pos2)
    d = math.sqrt(rpos[0]**2 + rpos[1]**2 + rpos[2]**2 + rpos[3]**2 +rpos[4]**2 + rpos[5]**2)
    return d

def CloseGap(taa1, taa2, delt):
    xconst = np.zeros((6, 1))
    for i in range(6):
        xconst[i] = taa2.TAA[i] - taa1.TAA[i]
    #normalize
    xret = np.zeros((6, 1))
    var = math.sqrt(xconst[0]**2 + xconst[1]**2 + xconst[2]**2)
    #print(var, "var")
    if var == 0:
        var = 0
    for i in range(6):
        xret[i] = taa1.TAA[i] + (xconst[i] / var) * delt
    #xf = tm1 @ TAAtoTM(xret)

    return tm(xret)

def ArcGap(taa1, taa2, delt):
    """Short summary.

    Args:
        taa1 (type): Description of parameter `taa1`.
        taa2 (type): Description of parameter `taa2`.
        delt (type): Description of parameter `delt`.

    Returns:
        type: Description of returned object.

    """
    xconst = np.zeros((6, 1))
    for i in range(6):
        xconst[i] = taa2[i] - taa1[i]
    #normalize
    xret = np.zeros((6, 1))
    var = math.sqrt(xconst[0]**2 + xconst[1]**2 + xconst[2]**2)
    #print(var, "var")
    if var == 0:
        var = 0
    for i in range(6):
        xret[i] = (xconst[i] / var) * delt
    xf = taa1 @ TAAtoTM(xret)

    return xf

def RotFromVec(pos1, pos2):
    """Short summary.

    Args:
        pos1 (type): Description of parameter `pos1`.
        pos2 (type): Description of parameter `pos2`.

    Returns:
        type: Description of returned object.

    """
    d = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2)
    res = lambda x : Distance(tm([pos1[0], pos1[1], pos1[2], x[0], x[1], pos1[5]]) @ tm([0, 0, d, 0, 0, 0]), pos2)
    x0 = np.array([pos1[3], pos1[4]])
    xs = sci.optimize.fmin(res, x0, disp=False)
    pos1[3:5] = xs.reshape((2))
    return pos1

def lookat(start, goal):
    """Short summary.

    Args:
        start (type): Description of parameter `start`.
        goal (type): Description of parameter `goal`.

    Returns:
        type: Description of returned object.

    """
    upa = (start @ tm([-1, 0, 0, 0, 0, 0]))
    up = upa[0:3].flatten()
    va = start[0:3].flatten()
    vb = goal[0:3].flatten()
    zax = mr.Normalize(vb-va)
    xax = mr.Normalize(np.cross(up, zax))
    yax = np.cross(zax, xax)
    R2 = np.eye(4)
    R2[0:3, 0:3] = np.array([xax, yax, zax]).T
    R2[0:3, 3] = va
    ttm = tm(R2)
    return ttm

def RotFromVec2(pos1, pos2):
    """Short summary.

    Args:
        pos1 (type): Description of parameter `pos1`.
        pos2 (type): Description of parameter `pos2`.

    Returns:
        type: Description of returned object.

    """
    d = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    #d = GlobalToLocal(pos1, pos2)[0]
    res = lambda x : Distance(pos1@ tm([0, 0, 0, 0, 0, x[0]]) @ tm([0, -d, 0, 0, 0, 0]), pos2)
    x0 = np.array(0)
    xs = sci.optimize.fmin(res, x0, disp=False)
    pos1 = pos1@ tm([0, 0, 0, 0, 0, round(xs[0], 1)])
    #pos1 = pos1 @ tm([0, 0, 0, 0, 0, np.pi/4])
    #G2 - 0
    #G4 - np.pi(?)
    #G3 - pi/2
    print(xs[0])
    return pos1

def IKPath(initial, goal, steps):
    """Short summary.

    Args:
        initial (type): Description of parameter `initial`.
        goal (type): Description of parameter `goal`.
        steps (type): Description of parameter `steps`.

    Returns:
        type: Description of returned object.

    """
    delta = (goal.gTAA() - initial.gTAA())/steps
    poslist = []
    for i in range(steps):
        pos = tm(inital.gTAA() + delta * i)
        poslist.append(pos)
    return poslist

#ANGLE HELPERS

def Deg2Rad(deg):
    """Short summary.

    Args:
        deg (type): Description of parameter `deg`.

    Returns:
        type: Description of returned object.

    """
    return deg * np.pi / 180

def Rad2Deg(rad):
    """Short summary.

    Args:
        rad (type): Description of parameter `rad`.

    Returns:
        type: Description of returned object.

    """
    return rad * 180 / np.pi

def AngleMod(rad):
    """Short summary.

    Args:
        rad (type): Description of parameter `rad`.

    Returns:
        type: Description of returned object.

    """
    if isinstance(rad, tm):
        return rad.AngleMod();
    if np.size(rad) == 1:
        if abs(rad) > 2 * np.pi:
            rad = rad % (2 * np.pi)
        return rad
    if np.size(rad) == 6:
        for i in range(3, 6):
            if abs(rad[i]) > 2 * np.pi:
                rad[i] = rad[i] % (2 * np.pi)
        return rad
    for i in range(np.size(rad)):
        if abs(rad[i]) > 2 * np.pi:
            rad[i] = rad[i] % (2 * np.pi)
    return rad

def AngleBetween(p1, p2, p3):
    """Short summary.

    Args:
        p1 (type): Description of parameter `p1`.
        p2 (type): Description of parameter `p2`.
        p3 (type): Description of parameter `p3`.

    Returns:
        type: Description of returned object.

    """
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1], p1[2] - p2[2]])
    #v1n = mr.Normalize(v1)
    v1n = np.linalg.norm(v1)
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1], p3[2] - p2[2]])
    #v2n = mr.Normalize(v2)
    v2n = np.linalg.norm(v2)
    res = np.clip(np.dot(v1, v2)/(v1n*v2n), -1, 1)
    #res = np.clip(np.dot(np.squeeze(v1n), np.squeeze(v2n)), -1, 1)
    res = AngleMod(math.acos(res))
    return res

def AngleBetweenXY(p1, p2, p3):
    """Short summary.

    Args:
        p1 (type): Description of parameter `p1`.
        p2 (type): Description of parameter `p2`.
        p3 (type): Description of parameter `p3`.

    Returns:
        type: Description of returned object.

    """
    v1 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
    v1m = np.sqrt(v1[0] **2 + v1[1]**2)
    v2 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
    v2m = np.sqrt(v2[0] **2 + v2[1]**2)
    dot = np.dot(v1, v2)
    res = np.arccos(dot/(v1m * v2m))
    return res

#Misc
def GenForceWrench(loc, force, forcedir):
    """
    Generates a new wrench
    Args:
        loc: relative location of wrench
        force: magnitude of force applied (or mass if forcedir is a gravity vector)
        forcedir: unit vector to apply force (or gravity)
    Returns:
        wrench
    """
    forcev = np.array(forcedir) * force #Force vector (negative Z)
    t_wren = np.cross(loc[0:3].reshape((3)), forcev) #Calculate moment based on position and action
    wrench = np.array([t_wren[0], t_wren[1], t_wren[2], forcev[0], forcev[1], forcev[2]]).reshape((6, 1)) #Create Complete Wrench
    return wrench

def TransformWrenchFrame(wrench, wrenchFrame, newFrame):
    """
    Translates one wrench frame to another
    Args:
        wrench: original wrench to be translated
        wrenchFrame: the original frame that the wrench was in (tm)
        newFrame: the new frame that the wrench *should* be in (tm)
    Returns:
        newWrench in the frame of newFrame
    """
    ref = GlobalToLocal(wrenchFrame, newFrame)
    return  ref.Adjoint().T @ wrench

def BoxSpatialInertia(m, l, w, h):
    """Short summary.

    Args:
        m (type): Description of parameter `m`.
        l (type): Description of parameter `l`.
        w (type): Description of parameter `w`.
        h (type): Description of parameter `h`.

    Returns:
        type: Description of returned object.

    """
    Ixx = m*(w*w+h*h)/12
    Iyy = m*(l*l+h*h)/12
    Izz = m*(w*w+l*l)/12
    Ib = np.diag((Ixx, Iyy, Izz))

    Gbox = np.vstack((np.hstack((Ib, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), m*np.identity((3))))))
    return Gbox

def unitSphere(num_points):
    """
    Generates a "unit sphere" with an approximate number of points
    numActual = round(num_points)^2
    Args:
        num_points: Approximate number of points to collect
    Returns:
        xyzcoords: points in cartesian coordinates
        azel: coords in azimuth/elevation notation
    """
    xyzcoords = []
    azel = []
    azr = round(math.sqrt(num_points))
    elr = round(math.sqrt(num_points))
    inc = np.pi * 2 / azr
    incb = 2 / elr
    a = 0
    e = -1
    for i in range(azr + 1):
        for j in range(elr + 1):
            x = np.cos(a) * np.sin(np.arccos(e))
            y = np.sin(a) * np.sin(np.arccos(e))
            z = np.cos(np.arccos(e))
            xyzcoords.append([x, y, z])
            azel.append([a, e])

            a = a + inc
        e = e + incb
        a = -1

    return xyzcoords, azel

def TwistToScrew(S):
    """Short summary.

    Args:
        S (type): Description of parameter `S`.

    Returns:
        type: Description of returned object.

    """
    if (Norm(S[0:3])) == 0:
        w = mr.Normalize(S[0:6])
        th = Norm(S[3:6])
        q = np.array([0, 0, 0]).reshape((3, 1))
        h = inf
    else:
        unitS = S/Norm(S[0:3])
        w = unitS[0:3].reshape((3))
        v = unitS[3:6].reshape((3))
        th = Norm(S[0:3])
        q = np.cross(w, v)
        h = (v.reshape((3, 1)) @ w.reshape((1, 3)))
    return (w, th, q, h)

def Norm(V):
    """Short summary.

    Args:
        V (type): Description of parameter `V`.

    Returns:
        type: Description of returned object.

    """
    C = np.linalg.norm(V)
    return C

def NormalizeTwist(tw):
    """Short summary.

    Args:
        tw (type): Description of parameter `tw`.

    Returns:
        type: Description of returned object.

    """
    if Norm(tw[0:3]) > 0:
        twn = tw/Norm(tw[0:3])
    else:
        twn = tw/norm(tw[3:6])
    return twn

def TwistFromTransform(tm):
    """Short summary.

    Args:
        tm (type): Description of parameter `tm`.

    Returns:
        type: Description of returned object.

    """
    tmskew = mr.MatrixLog6(tm.TM)
    return mr.se3ToVec(tmskew)

def TransformFromTwist(tw):
    """Short summary.

    Args:
        tw (type): Description of parameter `tw`.

    Returns:
        type: Description of returned object.

    """
    tw = tw.reshape((6))
    #print(tw)
    tms = mr.VecTose3(tw)
    tms = delMini(tms)
    tmr = mr.MatrixExp6(tms)
    return tm(tmr)

#def RotationAroundVector(w, theta):
#    r = np.identity(3)+math.sin(theta) * rp.skew(w)+(1-math.cos(theta)) * rp.skew(w) @ rp.skew(w)
#    return r

def SetElements(data, inds, vals):
    """Short summary.

    Args:
        data (type): Description of parameter `data`.
        inds (type): Description of parameter `inds`.
        vals (type): Description of parameter `vals`.

    Returns:
        type: Description of returned object.

    """
    res = copy.copy(data)
    for i in range(len(inds)):
        res[inds[i]] = vals[i]
    #res[inds] = vals
    #for i in range (0,(inds.size-1)):
    #    res[inds[i]] = vals[i]
    return res

def EKF(meant_1, covt_1, ctrlt, meast, gfn, hfn, dgdx, dgdu, dhdx, dhdv):
    """
    Extended Kalman Filter Iteration
    Args:
        meant_1:
        covt_1:
        ctrlt:
        meast:
        gfn:
        hfn:
        dgdx:
        dgdu:
        dhdx:
        dhdv:
    Returns:
        meant:
        covt:
    """
    predmeant = gfn(ctrlt, meant_1)
    Gt = dgdx(ctrlt, meant_1)
    Rt = dgdu(ctrlt, meant_1)
    predcovt = Gt @ covt_1 @ Gt.conj().T + Rt
    predmeast = hfn(predmeant)
    Ht = dhdx(predmeant)
    Qt = dhdv(predmeant)
    Kt = predcovt @ Ht.conj().T @ ling.inv(Ht @ predcovt @ Ht.conj().T+Qt)
    meant = predmeant + Kt @ (meast - predmeast)
    covt = (np.identity((Kt*Ht).shape[0])-Kt @ Ht) @ predcovt

    return meant, covt

#Jacobians
def ChainJacobian(screws, theta):
    """
    Chain Jacobian
    Args:
        Screws: screw list
        theta: theta to evaluate at
    Returns:
        jac: chain jacobian
    """
    jac = np.zeros((6, np.size(theta)))
    T = np.eye(4)
    Jac[0:6, 0] = screws[0:6, 0]

    for i in range(1, np.size(theta)):
        T = T * TransformFromTwist(theta[i-1]*screws[1:6, i-1])
        jac[0:6, i] = mr.Adjoint(T)*screws[0:6, i]
    return jac

def delMini(arr):
    """
    Deletes subarrays of dimension 1
    Requires 2d array
    Args:
        arr: array to prune
    Returns:
        newarr: pruned array
    """

    s = arr.shape
    newarr = np.zeros((s))
    for i in range(s[0]):
        for j in range(s[1]):
            newarr[i, j] = arr[i, j]
    return newarr


def NumJac(f, x0, h):
    """
    Calculates a numerical jacobian
    Args:
        f: function handle (FK)
        x0: initial value
        h: delta value
    Returns:
        dfdx: numerical Jacobian
    """
    x0p = np.copy(x0)
    x0p[0] = x0p[0] + h
    x0m = np.copy(x0)
    x0m[0] = x0m[0] - h
    dfdx = (f(x0p)-f(x0m))/(2*h)

    for i in range(1, x0.size):
        x0p =  np.copy(x0)
        x0p[i] = x0p[i] + h
        x0m =  np.copy(x0)
        x0m[i] = x0m[i] - h
        #Conversion paused here. continue evalutation
        dfdx=np.concatenate((dfdx,(f(x0p)-f(x0m))/(2*h)), axis = 0)
    dfdx=dfdx.conj().T
    f(x0)

    # Call the function with the initial input to reset state, if
    # applicable.
    #f(x0)

    return dfdx


def getUnitVec(point_1, point_2, distance = 1.0):
    """
    Returns a unit vector for a given actuator
    Args:
        point_1 (tm): Description of parameter `point_1`.
        point_2 (tm): Description of parameter `point_2`.
        distance (Float): Description of parameter `distance`.

    Returns:
        type: Description of returned object.

    """
    v1 = np.array([point_1[0], point_1[1], point_1[2]])
    unit_b = (np.array([point_2[0], point_2[1], point_2[2]]) - v1)
    unit = unit_b / ling.norm(unit_b)
    pos = v1 + (unit * distance)
    return tm([pos[0], pos[1], pos[2], 0, 0, 0])
