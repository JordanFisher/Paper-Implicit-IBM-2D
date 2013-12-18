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
Nb = 8*128
h = 1. / N
_h = 1. / _N
hb = 1. / Nb
dt = .00025 * h
#dt = .00001 * h
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

Ellipse (Nb, X, Y, .4, .2, .5, .5)


Old_X, Old_Y = 1. * X, 1. * Y

b = zeros(2*Nb,float64)
count = 0
TotalTime = 0
while CurT < T:
    Time = time.time()
    
    Xss, Yss = FiberForce (Nb, hb, s, X, Y, Xss, Yss)
    fx, fy = ForceToGrid (N, N, h, Nb, hb, X, Y, Xss, Yss)
    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)

    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)

    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, 1., X, Y, u, v)
    X += dt * Xvel
    Y += dt * Yvel
    
####    print time.time() - Time
####
####    MaxVel = 0
####    for i in range(N):
####        MaxVel = max(MaxVel, max((u[i,:]**2+v[i,:]**2)**.5))
####    print "MaxVel--------->", MaxVel
####
####    count += 1
####    if count % 20 == 0:
####        P = ifft2(P).real
####        clf()
####        imshow(u)
####        colorbar()
####        scatter(N*Y,N*X)
####        quiver(N*Y,N*X,Y-Old_Y,-(X-Old_X))
#####    raw_input("")


    Time = time.time() - Time
    TotalTime += Time
    count += 1

    Ts = [.0001,.0002,.0005,.001,.005]
    for _T in Ts:
        if CurT < _T and CurT + dt >= _T:
            print "Time", _T, ":", TotalTime, TotalTime / count
    print "Time:", CurT
    CurT += dt

