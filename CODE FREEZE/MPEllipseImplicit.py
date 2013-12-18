import sys
sys.path.append("c:\HeartValve")
sys.path.append("c:\HeartValve\IB_c\Release")
from scipy.linalg import eig as Eig
from pylab import *
from numpy import *
from numpy.fft import fft2, ifft2, fft as FFT, ifft as IFFT, rfft
from IB_Methods import *
import IB_c
from scipy import sparse, linsolve
import time
import math
import locale
#show()

def EzSparseScalarMult(EzA, a):
    N, M, MaxL, IShell, IShelli, IShelld = EzA

    return (N, M, MaxL, a * IShell, IShelli, IShelld)

def EzSparse(N,M,A):
    IShell = zeros((N,6),float64)
    IShelli = zeros((N,6),int32)
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


DeltaType = 0
N = 64 #4*128
_N = N
Nb = 2*N #2*128
h = 1. / N
_h = 1. / _N
hb = 1. / Nb
T = 1.
NT = 80
dt = T / NT
CurT = 0
mu = .05
rho = 1.
s = 1.

print "h:",h, ", dt:", dt


WideLambda = zeros((N,N),float64)
ShortLambda = zeros((N,N),float64)
IB_c.InitWideLaplacian(N, N, h, WideLambda)
IB_c.InitShortLaplacian(N, N, h, ShortLambda)
DxSymbol = InitDxSymbol(N, N, h)
DySymbol = InitDySymbol(N, N, h)

_h = h
_N, _M = int(1. / _h), int(1. / _h)
UField, VField, UField2, VField2 = InitVelField(_N, _M, _h, h, dt, rho, mu, DeltaType)

print UField[0,0],VField2[0,0]

X = zeros(Nb, float64)
Y = zeros(Nb, float64)
Xs = zeros(Nb, float64)
Ys = zeros(Nb, float64)
Xss = zeros(Nb, float64)
Yss = zeros(Nb, float64)
Norm = zeros(Nb, float64)
F = zeros(Nb, float64)
G = zeros(Nb, float64)

u = zeros((N,N), float64)
v = zeros((N,N), float64)
ux = zeros((N,N), float64)
vx = zeros((N,N), float64)
uy = zeros((N,N), float64)
vy = zeros((N,N), float64)
fx = zeros((N,N), float64)
fy = zeros((N,N), float64)




def FiberForce (Nb, hb, s, X, Y):
    for i in range(Nb):
        Xs[i] = (X[(i+1)%Nb] - X[i]) / hb
        Ys[i] = (Y[(i+1)%Nb] - Y[i]) / hb
        Xss[i] = (X[(i+1)%Nb] - 2 * X[i] + X[(i-1)%Nb]) / hb**2
        Yss[i] = (Y[(i+1)%Nb] - 2 * Y[i] + Y[(i-1)%Nb]) / hb**2
        Norm[i] = (Xs[i]**2 + Ys[i]**2)**.5

    for i in range(Nb):
        F[i] = Xss[i] + (Xs[i] * Norm[i] - Xs[(i-1)%Nb] * Norm[(i-1)%Nb]) / hb
        G[i] = Yss[i] + (Ys[i] * Norm[i] - Ys[(i-1)%Nb] * Norm[(i-1)%Nb]) / hb

    return F, G

def Jacobian (Nb, hb, s, X, Y, Xs, Ys, J):
    N2 = 1. / hb**2
    N1 = 1. / hb
    
    for i in range(Nb):
        J[i, i] = N2 * (-2 - Norm[i] - Xs[i]**2 / Norm[i] - Norm[(i-1)%Nb] - Xs[(i-1)%Nb]**2 / Norm[(i-1)%Nb])
        J[i, i+Nb] = -N2 * (Xs[i] * Ys[i] / Norm[i] + Xs[(i-1)%Nb] * Ys[(i-1)%Nb] / Norm[(i-1)%Nb])
        J[i, (i+1)%Nb] = N2 * (1 + Norm[i] + Xs[i]**2 / Norm[i])
        J[i, (i+1)%Nb+Nb] = N2 * (Xs[i] * Ys[i] / Norm[i])
        J[i, (i-1)%Nb] = N2 * (1 + Norm[(i-1)%Nb] + Xs[(i-1)%Nb]**2 / Norm[(i-1)%Nb])
        J[i, (i-1)%Nb+Nb] = N2 * (Xs[(i-1)%Nb] * Ys[(i-1)%Nb] / Norm[(i-1)%Nb])
        
        J[i+Nb, i+Nb] = N2 * (-2 - Norm[i] - Ys[i]**2 / Norm[i] - Norm[(i-1)%Nb] - Ys[(i-1)%Nb]**2 / Norm[(i-1)%Nb])
        J[i+Nb, i] = -N2 * (Xs[i] * Ys[i] / Norm[i] + Xs[(i-1)%Nb] * Ys[(i-1)%Nb] / Norm[(i-1)%Nb])
        J[i+Nb, (i+1)%Nb+Nb] = N2 * (1 + Norm[i] + Ys[i]**2 / Norm[i])
        J[i+Nb, (i+1)%Nb] = N2 * (Xs[i] * Ys[i] / Norm[i])
        J[i+Nb, (i-1)%Nb+Nb] = N2 * (1 + Norm[(i-1)%Nb] + Ys[(i-1)%Nb]**2 / Norm[(i-1)%Nb])
        J[i+Nb, (i-1)%Nb] = N2 * (Xs[(i-1)%Nb] * Ys[(i-1)%Nb] / Norm[(i-1)%Nb])

def SeedJacobian (Nb):
    J = zeros((2*Nb,2*Nb),float64)
    
    for i in range(Nb):
        J[i, i] = i + 1
        J[i, i+Nb] = i + Nb + 1
        J[i, (i+1)%Nb] = i + 2*Nb + 1
        J[i, (i+1)%Nb+Nb] = i + 3*Nb + 1
        J[i, (i-1)%Nb] = i + 4*Nb + 1
        J[i, (i-1)%Nb+Nb] = i + 5*Nb + 1
        
        J[i+Nb, i+Nb] = i + 6*Nb + 1
        J[i+Nb, i] = i + 7*Nb + 1
        J[i+Nb, (i+1)%Nb+Nb] = i + 8*Nb + 1
        J[i+Nb, (i+1)%Nb] = i + 9*Nb + 1
        J[i+Nb, (i-1)%Nb+Nb] = i + 10*Nb + 1
        J[i+Nb, (i-1)%Nb] = i + 11*Nb + 1

    return sparse.csr_matrix(J)

def SeedEzJacobian (Nb, J):
    N, M, MaxL, IShell, IShelli, IShelld = J
    
    for i in range(Nb):
        IShelli[i,0] = i
        IShelli[i,1] = i + Nb
        IShelli[i,2] = (i + 1) % Nb
        IShelli[i,3] = (i + 1) % Nb + Nb
        IShelli[i,4] = (i - 1) % Nb
        IShelli[i,5] = (i - 1) % Nb + Nb
        
        IShelli[i+Nb,1] = i
        IShelli[i+Nb,0] = i + Nb
        IShelli[i+Nb,3] = (i + 1) % Nb
        IShelli[i+Nb,2] = (i + 1) % Nb + Nb
        IShelli[i+Nb,5] = (i - 1) % Nb
        IShelli[i+Nb,4] = (i - 1) % Nb + Nb
    

def JacobianFast (Nb, hb, s, X, Y, Xs, Ys, J, args):
    N2 = 1. / hb**2
    N1 = 1. / hb

    Vals = 0. * J.data

    Vals[0] = N2 * (-2 - Norm[0] - Xs[0]**2 / Norm[0] - Norm[(0-1)%Nb] - Xs[(0-1)%Nb]**2 / Norm[(0-1)%Nb])
    Vals[1:Nb] = N2 * (-2 - Norm[1:] - Xs[1:]**2 / Norm[1:] - Norm[0:Nb-1] - Xs[0:Nb-1]**2 / Norm[0:Nb-1])

    Vals[Nb] = -N2 * (Xs[0] * Ys[0] / Norm[0] + Xs[(0-1)%Nb] * Ys[(0-1)%Nb] / Norm[(0-1)%Nb])
    Vals[Nb+1:2*Nb] = -N2 * (Xs[1:] * Ys[1:] / Norm[1:] + Xs[:Nb-1] * Ys[:Nb-1] / Norm[:Nb-1])
        
    Vals[2*Nb] = N2 * (1 + Norm[0] + Xs[0]**2 / Norm[0])
    Vals[2*Nb+1:3*Nb] = N2 * (1 + Norm[1:] + Xs[1:]**2 / Norm[1:])

    Vals[3*Nb] = N2 * (Xs[0] * Ys[0] / Norm[0])
    Vals[3*Nb+1:4*Nb] = N2 * (Xs[1:] * Ys[1:] / Norm[1:])
    
    Vals[4*Nb] = N2 * (1 + Norm[(0-1)%Nb] + Xs[(0-1)%Nb]**2 / Norm[(0-1)%Nb])
    Vals[4*Nb+1:5*Nb] = N2 * (1 + Norm[:Nb-1] + Xs[:Nb-1]**2 / Norm[:Nb-1])
    
    Vals[5*Nb] = N2 * (Xs[(0-1)%Nb] * Ys[(0-1)%Nb] / Norm[(0-1)%Nb])
    Vals[5*Nb+1:6*Nb] = N2 * (Xs[:Nb-1] * Ys[:Nb-1] / Norm[:Nb-1])




    Vals[6*Nb] = N2 * (-2 - Norm[0] - Ys[0]**2 / Norm[0] - Norm[(0-1)%Nb] - Ys[(0-1)%Nb]**2 / Norm[(0-1)%Nb])
    Vals[6*Nb+1:7*Nb] = N2 * (-2 - Norm[1:] - Ys[1:]**2 / Norm[1:] - Norm[0:Nb-1] - Ys[0:Nb-1]**2 / Norm[0:Nb-1])

    Vals[7*Nb] = -N2 * (Xs[0] * Ys[0] / Norm[0] + Xs[(0-1)%Nb] * Ys[(0-1)%Nb] / Norm[(0-1)%Nb])
    Vals[7*Nb+1:8*Nb] = -N2 * (Xs[1:] * Ys[1:] / Norm[1:] + Xs[:Nb-1] * Ys[:Nb-1] / Norm[:Nb-1])
        
    Vals[8*Nb] = N2 * (1 + Norm[0] + Ys[0]**2 / Norm[0])
    Vals[8*Nb+1:9*Nb] = N2 * (1 + Norm[1:] + Ys[1:]**2 / Norm[1:])

    Vals[9*Nb] = N2 * (Xs[0] * Ys[0] / Norm[0])
    Vals[9*Nb+1:10*Nb] = N2 * (Xs[1:] * Ys[1:] / Norm[1:])
    
    Vals[10*Nb] = N2 * (1 + Norm[(0-1)%Nb] + Ys[(0-1)%Nb]**2 / Norm[(0-1)%Nb])
    Vals[10*Nb+1:11*Nb] = N2 * (1 + Norm[:Nb-1] + Ys[:Nb-1]**2 / Norm[:Nb-1])
    
    Vals[11*Nb] = N2 * (Xs[(0-1)%Nb] * Ys[(0-1)%Nb] / Norm[(0-1)%Nb])
    Vals[11*Nb+1:12*Nb] = N2 * (Xs[:Nb-1] * Ys[:Nb-1] / Norm[:Nb-1])

    J.data[args] = Vals

def JacobianFast2 (Nb, hb, s, X, Y, Xs, Ys, J):
    N2 = 1. / hb**2
    N1 = 1. / hb

    N, M, MaxL, IShell, IShelli, IShelld = J

    
    IShell[0,0] = N2 * (-2 - Norm[0] - Xs[0]**2 / Norm[0] - Norm[(0-1)%Nb] - Xs[(0-1)%Nb]**2 / Norm[(0-1)%Nb])
    IShell[1:Nb,0] = N2 * (-2 - Norm[1:] - Xs[1:]**2 / Norm[1:] - Norm[0:Nb-1] - Xs[0:Nb-1]**2 / Norm[0:Nb-1])

    IShell[0,1] = -N2 * (Xs[0] * Ys[0] / Norm[0] + Xs[(0-1)%Nb] * Ys[(0-1)%Nb] / Norm[(0-1)%Nb])
    IShell[1:Nb,1] = -N2 * (Xs[1:] * Ys[1:] / Norm[1:] + Xs[:Nb-1] * Ys[:Nb-1] / Norm[:Nb-1])
        
    IShell[0,2] = N2 * (1 + Norm[0] + Xs[0]**2 / Norm[0])
    IShell[1:Nb,2] = N2 * (1 + Norm[1:] + Xs[1:]**2 / Norm[1:])

    IShell[0,3] = N2 * (Xs[0] * Ys[0] / Norm[0])
    IShell[1:Nb,3] = N2 * (Xs[1:] * Ys[1:] / Norm[1:])
    
    IShell[0,4] = N2 * (1 + Norm[(0-1)%Nb] + Xs[(0-1)%Nb]**2 / Norm[(0-1)%Nb])
    IShell[1:Nb,4] = N2 * (1 + Norm[:Nb-1] + Xs[:Nb-1]**2 / Norm[:Nb-1])
    
    IShell[0,5] = N2 * (Xs[(0-1)%Nb] * Ys[(0-1)%Nb] / Norm[(0-1)%Nb])
    IShell[1:Nb,5] = N2 * (Xs[:Nb-1] * Ys[:Nb-1] / Norm[:Nb-1])




    IShell[Nb,0] = N2 * (-2 - Norm[0] - Ys[0]**2 / Norm[0] - Norm[(0-1)%Nb] - Ys[(0-1)%Nb]**2 / Norm[(0-1)%Nb])
    IShell[Nb+1:,0] = N2 * (-2 - Norm[1:] - Ys[1:]**2 / Norm[1:] - Norm[0:Nb-1] - Ys[0:Nb-1]**2 / Norm[0:Nb-1])

    IShell[Nb,1] = -N2 * (Xs[0] * Ys[0] / Norm[0] + Xs[(0-1)%Nb] * Ys[(0-1)%Nb] / Norm[(0-1)%Nb])
    IShell[Nb+1:,1] = -N2 * (Xs[1:] * Ys[1:] / Norm[1:] + Xs[:Nb-1] * Ys[:Nb-1] / Norm[:Nb-1])
        
    IShell[Nb,2] = N2 * (1 + Norm[0] + Ys[0]**2 / Norm[0])
    IShell[Nb+1:,2] = N2 * (1 + Norm[1:] + Ys[1:]**2 / Norm[1:])

    IShell[Nb,3] = N2 * (Xs[0] * Ys[0] / Norm[0])
    IShell[Nb+1:,3] = N2 * (Xs[1:] * Ys[1:] / Norm[1:])
    
    IShell[Nb,4] = N2 * (1 + Norm[(0-1)%Nb] + Ys[(0-1)%Nb]**2 / Norm[(0-1)%Nb])
    IShell[Nb+1:,4] = N2 * (1 + Norm[:Nb-1] + Ys[:Nb-1]**2 / Norm[:Nb-1])
    
    IShell[Nb,5] = N2 * (Xs[(0-1)%Nb] * Ys[(0-1)%Nb] / Norm[(0-1)%Nb])
    IShell[Nb+1:,5] = N2 * (Xs[:Nb-1] * Ys[:Nb-1] / Norm[:Nb-1])

    

Ellipse (Nb, X, Y, 1/3., 1/4., .5, .5)
##CrazyEllipse(Nb, X, Y, .4, .2, .5, .5, .04, 5)
clf()
scatter(X,Y)


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
   

        


JSeed = SeedJacobian(Nb)
SeedArgs = argsort(JSeed.data)
J2 = 0. * JSeed
Diag = range(2*Nb)
EzJ = EzSparse(2*Nb,2*Nb,JSeed.todense())
SeedEzJacobian(Nb, EzJ)


B = zeros((2*Nb,2*Nb), float64)
diag = range(2*Nb)

Old_X, Old_Y = 1. * X, 1. * Y
y = zeros(2*Nb,float64)

b = zeros(2*Nb,float64)
count = 0
TotalTime = 0
ShowStats = False
Times = []
while CurT < T:
    Time = time.time()

    Old_u, Old_v = 1. * u, 1. * v
    Old_X, Old_Y = 1. * X, 1. * Y

    IB_c.CentralDerivative_x (N, N, h, u, ux)
    IB_c.CentralDerivative_x (N, N, h, v, vx)
    IB_c.CentralDerivative_y (N, N, h, u, uy)
    IB_c.CentralDerivative_y (N, N, h, v, vy)

    fx, fy = ExplicitTerms (dt, u, v, 0., 0., 0., 0., 0., 0., rho)
    #fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, 0., 0., rho)
    
    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy, rho, mu)
    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, 1., X, Y, u, v, DeltaType)
    if ShowStats:
        print "Explicit:", time.time() - Time

    A = zeros((2*Nb,2*Nb),float64)
    IB_c.ComputeMatrix (X, Y, _N, _N, Nb, _h, hb, UField, VField, UField2, VField2, A, -1)#, band = Nb/8+1)

    x = zeros(2*Nb,float64)
    J = zeros((2*Nb,2*Nb),float64)
    x[:Nb], x[Nb:] = X, Y
    f = 0. * x
    b[:Nb], b[Nb:] = dt * Xvel, dt * Yvel
    b += 1. * x

    Res = 10
    while Res > .0001:
    #for qq in range(2):

        holdtime = time.time()        
        
        f[:Nb], f[Nb:] = FiberForce(Nb, hb, s, x[:Nb], x[Nb:])


        JacobianFast2(Nb, hb, s, x[:Nb], x[Nb:], Xs, Ys, EzJ)
        EzJ = EzSparseScalarMult(EzJ, -dt * s)
        P = EzSparseMM (EzJ, A).T
        P[Diag,Diag] += ones(2*Nb)

##        print "P and J:", time.time() - holdtime
        

        r = b - (x - dt * s * dot(A, f))
        Res = max(abs(r))
##        print "Res:", Res
######        x += linalg.solve(P, r)

        __P = [P]
        for i in range(len(I)):
            __P.append(EzSparseMM(IspT[i],EzSparseMM(IspT[i], __P[i].T).T) )
#            __P.append(array( (I[i].T * matrix(__P[i]) * I[i]).todense() ))


        y = 0. * x
        for zz in range(3):
            y = FastMultigridStep(__P,r,I,y,.5)
        x += y


#    X, Y = X + y[0:Nb], Y + y[Nb:2*Nb]
    X, Y = x[0:Nb], x[Nb:2*Nb]
#    scatter(X,Y,c='r'); raw_input("")




    u, v = 1. * Old_u, 1. * Old_v
    
    Xss, Yss = FiberForce (Nb, hb, s, X, Y)
    fx, fy = ForceToGrid (N, N, h, Nb, hb, Old_X, Old_Y, Xss, Yss, DeltaType)
    #fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy, rho)
    fx, fy = ExplicitTerms (dt, u, v, 0., 0., 0., 0., fx, fy, rho)

    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy, rho, mu)

    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, 1., X, Y, u, v, DeltaType)


    Time = time.time() - Time
    TotalTime += Time
    count += 1

    if count % 1 == 0:
        P = ifft2(P).real
        clf()
        #imshow(u)
        imshow((u**2+v**2)**.5)
        #imshow(uy - vx)
        colorbar()
        scatter(N*Y,N*X)
        quiver(N*Y,N*X,Y-Old_Y,-(X-Old_X))

##    Ts = [.001,.002,.005,.01,.05]
##    for _T in Ts:
##        if CurT < _T and CurT + dt >= _T:
##            print "Time", _T, ":", TotalTime, TotalTime / count
##            Times.append(TotalTime)
    print "Time:", CurT, Time
    CurT += dt

print TotalTime
print TotalTime / float(NT)

Times = [TotalTime / count] + Times

def number_format(num, places=0):
    return locale.format("%.*f", (places, num), True)

for num in Times:
    print number_format(num, 3),'&',
    
