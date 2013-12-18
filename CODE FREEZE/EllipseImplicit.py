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



N = 1*128
_N = 1*128
Nb = 2*128
h = 1. / N
_h = 1. / _N
hb = 1. / Nb
dt = .02 * h
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


I = []
Isp = []
IspT = []
__N = Nb
while __N > 2 and __N%2==0:
    _I = zeros((2*__N,__N), float64)
    for i in range(__N/2):
        _I[2*i,i] = 1.
        _I[2*i+1,(i+1)%(__N/2)] = .5
        _I[2*i+1,(i)%(__N/2)] = .5

        _I[2*i+__N,i+__N/2] = 1.
        _I[2*i+1+__N,(i+1)%(__N/2)+__N/2] = .5
        _I[2*i+1+__N,(i)%(__N/2)+__N/2] = .5

    I.append(sparse.csc_matrix(_I))
    Isp.append(EzSparse(2*__N,__N,_I))
    IspT.append(EzSparse(__N,2*__N,_I.T))
    __N /= 2
   

DssMatrix *= -dt * s
DssSp = EzSparse(2*Nb,2*Nb,DssMatrix)


        




B = zeros((2*Nb,2*Nb), float64)
diag = range(2*Nb)

Old_X, Old_Y = 1. * X, 1. * Y
y = zeros(2*Nb,float64)

b = zeros(2*Nb,float64)
count = 0
TotalTime = 0
ShowStats = True
while CurT < T:
    Time = time.time()

    Old_u, Old_v = 1. * u, 1. * v
    Old_X, Old_Y = 1. * X, 1. * Y

    IB_c.CentralDerivative_x (N, N, h, u, ux)
    IB_c.CentralDerivative_x (N, N, h, v, vx)
    IB_c.CentralDerivative_y (N, N, h, u, uy)
    IB_c.CentralDerivative_y (N, N, h, v, vy)

    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, 0., 0.)    
    
    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, 1., X, Y, u, v)
    if ShowStats:
        print "Explicit:", time.time() - Time
        Time = time.time()

    A = zeros((2*Nb,2*Nb),float64)
    IB_c.ComputeMatrix (X, Y, _N, _N, Nb, _h, hb, UField, VField, UField2, VField2, A, -1)#, band = Nb/8+1)
    if ShowStats:
        print "Construct M:", time.time() - Time
        Time = time.time()

    P = EzSparseMM(DssSp, A).T
    P[diag,diag] += 1.

    if ShowStats:
        print "Calculate P:", time.time() - Time
        Time = time.time()

    __P = [P]
    for i in range(len(I)):
#        print i
        __P.append( EzSparseMM(IspT[i],EzSparseMM(IspT[i], __P[i].T).T) )

    if ShowStats:
        print "Init MG:", time.time() - Time

  
    for i in range(Nb):
        b[i] = X[i] + dt * Xvel[i]
        b[i+Nb] = Y[i] + dt * Yvel[i]

    y[:Nb], y[Nb:] = 1. * X, 1. * Y
    MaxRes = 10000
    Multis = 0
    while MaxRes > 30*dt:
        Multis += 1
        if ShowStats:
            Time = time.time()
        y = FastMultigridStep(__P,b,I,y,.05)
        if ShowStats:
            print "MG:", time.time() - Time
        r = dot(P,y) - b
        MaxRes = max(list(abs(r)))
        if ShowStats:
            print MaxRes,':',.1*dt
#    y = linalg.solve(P,b)

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
#    raw_input("")

    Time = time.time() - Time
    TotalTime += Time
    count += 1

    Ts = [.0001,.0002,.0005,.001,.005]
    for _T in Ts:
        if CurT < _T and CurT + dt >= _T:
            print "Time", _T, ":", TotalTime, TotalTime / count
    print "Time:", CurT, Multis
    CurT += dt


