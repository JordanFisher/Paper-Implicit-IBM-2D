import sys
sys.path.append("c:\HeartValve")
sys.path.append("c:\HeartValve\IB_c\Release")
from pylab import *
from numpy import *
from numpy.fft import fft2, ifft2, fft, ifft, rfft
from IB_Methods import *
import IB_c
from ValveSetup import *
import time
#show()



def FiberForce (Nb, X, Y, Xss, Yss, Links, Tethers):
    for i in range(Nb):
        Xss[i] = Yss[i] = 0.
    
    for A in Links:
        a, b, s, L = A

        l = ((X[b] - X[a])**2 + (Y[b] - Y[a])**2)**.5
        tx, ty = (X[b] - X[a]) / l, (Y[b] - Y[a]) / l

        Xss[a] += s * (X[b] - X[a] - L * tx)
        Yss[a] += s * (Y[b] - Y[a] - L * ty)
        Xss[b] -= s * (X[b] - X[a] - L * tx)
        Yss[b] -= s * (Y[b] - Y[a] - L * ty)

    for A in Tethers:
        a, _x, s = A
        x, y = _x

        Xss[a] += s * (x - X[a])
        Yss[a] += s * (y - Y[a])

    return Xss, Yss
    
 


  


Domain_x, Domain_y = 2., 1.
h = .003#.007
h = Domain_x / 800.
N, M = int(Domain_x / h), int(Domain_y / h)
#dt = .0000065
dt = .0004
T = .05
CurT = 0

#_s = 5000000000.
_s = 500000
Current = 100.

WideLambda = zeros((N,M),float64)
ShortLambda = zeros((N,M),float64)
IB_c.InitWideLaplacian(N, M, h, WideLambda)
IB_c.InitShortLaplacian(N, M, h, ShortLambda)
DxSymbol = InitDxSymbol(N, M, h)
DySymbol = InitDySymbol(N, M, h)

#UField, VField = InitVelField(N, M, h, h, dt)

X, Y, Links, Tethers, Parts = ConstructValve (_s, 0., Domain_x, 0., Domain_y, .014, True,1.15,.8,False)#.01)
#X, Y, Links, Tethers, Parts = ConstructValve (_s, 0., Domain_x, 0., Domain_y, .012, True,1.5,1.)#.01)
#X, Y, Links, Tethers, Valve = ConstructValve (_s, 0., 2., 0., 1., .03)#.01)
Links = []
Nb = len(X)
hb = 1. / Nb
##for j in range(Nb):
##    X[j] += .002*cos(j)
##    Y[j] += .002*sin(j**2)

Xs = zeros(Nb, float64)
Ys = zeros(Nb, float64)
Xss = zeros(Nb, float64)
Yss = zeros(Nb, float64)
u = zeros((N,M), float64)
v = zeros((N,M), float64)
ux = zeros((N,M), float64)
vx = zeros((N,M), float64)
uy = zeros((N,M), float64)
vy = zeros((N,M), float64)
fx = zeros((N,M), float64)
fy = zeros((N,M), float64)


PlotValve (X, Y, Links)


Times = []
A = zeros((2*Nb,2*Nb), float64)


b = zeros(2*Nb,float64)
count = 0
while CurT < T:
    Old_X, Old_Y = 1. * X, 1. * Y
    print "Time:", CurT, count
    CurT += dt

    Time = time.time()

    IB_c.CentralDerivative_x (N, M, h, u, ux)
    IB_c.CentralDerivative_x (N, M, h, v, vx)
    IB_c.CentralDerivative_y (N, M, h, u, uy)
    IB_c.CentralDerivative_y (N, M, h, v, vy)

    Xss, Yss = FiberForce (Nb, X, Y, Xss, Yss, Links, Tethers)   
    fx, fy = ForceToGrid (N, M, h, Nb, hb, X, Y, Xss, Yss)
    
    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)
    fx += dt * Current * ones((N,M),float64)

    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)

    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, M, h, Nb, hb, 1., X, Y, u, v)
    X, Y = X + dt * Xvel, Y + dt * Yvel

    print "Explicit time:", time.time() - Time
    Times.append(time.time() - Time)
    print "avg:", sum(Times)/len(Times)

    count += 1
    if count % 1 == 0:
        P = ifft2(P).real
        clf()
        imshow(P)
        scatter(M*Y,N*X/2)
        quiver(M*Y,N*X/2,Yvel,-Xvel)
#        #_!_print gaussx[0:Nb]
#        raw_input("")
