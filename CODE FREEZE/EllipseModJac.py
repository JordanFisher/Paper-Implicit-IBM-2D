import sys
sys.path.append("c:\HeartValve")
sys.path.append("c:\HeartValve\IB_c\Release")
from pylab import *
from numpy import *
from numpy.fft import fft2, ifft2, fft as FFT, ifft as IFFT, rfft
from IB_Methods import *
import IB_c
from scipy import sparse, linsolve
import time
import math
#show()

def EzSparse(N,M,A):
    IShell = zeros((N,3),float64)
    IShelli = zeros((N,3),int32)
    IShelld = zeros(N,int32)

    IB_c.EzSparse(N,M,A,IShell,IShelli,IShelld)

    return (N, M, max(list(IShelld)), IShell, IShelli, IShelld)

def EzSparseMM(A1, A2, newB = True, B = -1):
    N, M, s, IShell, IShelli, IShelld = A1

    _N, _M = A2.shape

    if M != _N:
        raise("Incompatible dimensions")

    if newB:
        B = zeros((N,_M),float64)

    IB_c.SparseMM(N, M, _M, s, IShell, IShelli, IShelld, A2, B)

    return B


def FiberForce (Nb, hb, s, X, Y, Xss, Yss):
    SecondDerivativePeriodic (Nb, hb, X, Xss)
    SecondDerivativePeriodic (Nb, hb, Y, Yss)

    Xss, Yss = s * Xss, s * Yss

    return Xss, Yss

def ComputeDssMatrix (Nb, hb):
    DssMatrix = zeros((2*Nb,2*Nb), float64)
    DssMatrix[0,0] = -2. / hb**2
    DssMatrix[0,1] = 1. / hb**2
    DssMatrix[0,Nb-1] = 1. / hb**2
    for i in range(1,Nb-1):
        DssMatrix[i,i] = -2. / hb**2
        DssMatrix[i,i+1] = 1. / hb**2
        DssMatrix[i,i-1] = 1. / hb**2
    DssMatrix[Nb-1,Nb-1] = -2. / hb**2
    DssMatrix[Nb-1,Nb-2] = 1. / hb**2
    DssMatrix[Nb-1,0] = 1. / hb**2

    DssMatrix[Nb,Nb] = -2. / hb**2
    DssMatrix[Nb,Nb+1] = 1. / hb**2
    DssMatrix[Nb,2*Nb-1] = 1. / hb**2
    for i in range(1,Nb-1):
        DssMatrix[Nb+i,Nb+i] = -2. / hb**2
        DssMatrix[Nb+i,Nb+i+1] = 1. / hb**2
        DssMatrix[Nb+i,Nb+i-1] = 1. / hb**2
    DssMatrix[2*Nb-1,2*Nb-1] = -2. / hb**2
    DssMatrix[2*Nb-1,2*Nb-2] = 1. / hb**2
    DssMatrix[2*Nb-1,Nb] = 1. / hb**2

    return DssMatrix



N = 4*128
_N = 4*128
Nb = 4*128
h = 1. / N
_h = 1. / _N
hb = 1. / Nb
dt = .01 * h
#dt = .00003 * h
T = .005
CurT = 0
s = 100000.
#s = 10000000.

print "h:",h, ", dt:", dt


WideLambda = zeros((N,N),float64)
ShortLambda = zeros((N,N),float64)
IB_c.InitWideLaplacian(N, N, h, WideLambda)
IB_c.InitShortLaplacian(N, N, h, ShortLambda)
DxSymbol = InitDxSymbol(N, N, h)
DySymbol = InitDySymbol(N, N, h)

_h = h
_N, _M = int(1. / _h), int(1. / _h)
UField, VField, UField2, VField2 = InitVelField(_N, _M, _h, h, dt)

print UField[0,0],VField2[0,0]
clf()
clf(); imshow(UField); colorbar(); #show(); #&&raw_input("")
clf(); imshow(VField2); colorbar(); #show(); #&&raw_input("")


X = zeros(Nb, float64)
Y = zeros(Nb, float64)
Xss = zeros(Nb, float64)
Yss = zeros(Nb, float64)
u = zeros((N,N), float64)
v = zeros((N,N), float64)
ux = zeros((N,N), float64)
vx = zeros((N,N), float64)
uy = zeros((N,N), float64)
vy = zeros((N,N), float64)
fx = zeros((N,N), float64)
fy = zeros((N,N), float64)

DssMatrix = ComputeDssMatrix (Nb, hb)
Ellipse (Nb, X, Y, .4, .2, .5, .5)


DssMatrix *= -dt * s
DssSp = EzSparse(2*Nb,2*Nb,DssMatrix)


        




B = zeros((2*Nb,2*Nb), float64)
ADiag = zeros((2*Nb,2*Nb), float64)
diag = range(2*Nb)

Old_X, Old_Y = 1. * X, 1. * Y
y = zeros(2*Nb,float64)

b = zeros(2*Nb,float64)
count = 0
TotalTime = 0
ShowStats = True
while CurT < T:
    Old_u, Old_v = 1. * u, 1. * v
    Old_X, Old_Y = 1. * X, 1. * Y

    Time = time.time()
    
    IB_c.CentralDerivative_x (N, N, h, u, ux)
    IB_c.CentralDerivative_x (N, N, h, v, vx)
    IB_c.CentralDerivative_y (N, N, h, u, uy)
    IB_c.CentralDerivative_y (N, N, h, v, vy)

    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, 0., 0.)    

    print time.time() - Time; Time = time.time()
    
    fx = fft2(fx)
    fy = fft2(fy)
    print "FFT:", time.time() - Time; Time = time.time()    

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
    print time.time() - Time; Time = time.time()

    u = ifft2(u).real
    v = ifft2(v).real
    print "iFFT:", time.time() - Time; Time = time.time()    
    
    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, 1., X, Y, u, v)
    if ShowStats:
        print "Explicit:", time.time() - Time

    A = zeros((2*Nb,2*Nb),float64)
    IB_c.ComputeMatrix (X, Y, _N, _N, Nb, _h, hb, UField, VField, UField2, VField2, A, -1)#, band = Nb/8+1)

    ADiag[diag,diag] = A[diag,diag]

    P = EzSparseMM(DssSp, A).T
    P[diag,diag] += 1.
    PP = EzSparseMM(DssSp, ADiag).T
    PP[diag,diag] += 1.
#    PP2 = identity(2*Nb) + dot(DssMatrix, ADiag)

    #P = identity(2*Nb) + dot(A,DssMatrix)

    for i in range(Nb):
        b[i] = X[i] + dt * Xvel[i]
        b[i+Nb] = Y[i] + dt * Yvel[i]

##    y[:Nb], y[Nb:] = 1. * X, 1. * Y
##    MaxRes = 10000
##    Multis = 0
##    while MaxRes > 30*dt:
##        Multis += 1
##        y += .25 * linalg.solve(PP,b-dot(P,y))
##        r = dot(P,y) - b
##        MaxRes = max(list(abs(r)))
##        if ShowStats:
##            print MaxRes,':',30*dt
    Multis = 0
    y = linalg.solve(P,b)

#    X, Y = X + y[0:Nb], Y + y[Nb:2*Nb]
    X, Y = y[0:Nb], y[Nb:2*Nb]
#    scatter(X,Y,c='r'); raw_input("")




    u, v = 1. * Old_u, 1. * Old_v
    
    Xss, Yss = FiberForce (Nb, hb, s, X, Y, Xss, Yss)
    fx, fy = ForceToGrid (N, N, h, Nb, hb, Old_X, Old_Y, Xss, Yss)
    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)

    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)

    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, 1., X, Y, u, v)
##    print X - Old_X - dt*Xvel
##    print Y - Old_Y - dt*Yvel    

#    print time.time() - Time

##    MaxVel = 0
##    for i in range(N):
##        MaxVel = max(MaxVel, max((u[i,:]**2+v[i,:]**2)**.5))
##    print "MaxVel--------->", MaxVel


    if count % 1 == 0:
        P = ifft2(P).real
        clf()
        imshow(u)
        colorbar()
        scatter(N*Y,N*X)
        quiver(N*Y,N*X,Y-Old_Y,-(X-Old_X))
    raw_input("")

    Time = time.time() - Time
    TotalTime += Time
    count += 1

    Ts = [.0001,.0002,.0005,.001,.005]
    for _T in Ts:
        if CurT < _T and CurT + dt >= _T:
            print "Time", _T, ":", TotalTime, TotalTime / count
    print "Time:", CurT, Multis
    CurT += dt


