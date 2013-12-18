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



N = 1*128
Nb = 2*N #2*128
h = 1. / N
hb = 1. / Nb
dt = .001#.25 * .0015 / 2.
T = 1.
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
UField, VField, UField2, VField2 = InitVelField(_N, _M, _h, h, dt, rho, mu)

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


    

Ellipse (Nb, X, Y, 1/3., 1/4., .5, .5)
##CrazyEllipse(Nb, X, Y, .4, .2, .5, .5, .04, 5)
clf()
scatter(X,Y)



Old_X, Old_Y = 1. * X, 1. * Y

count = 0
TotalTime = 0
ShowStats = True
Times = []
while CurT < T:
    Time = time.time()

    IB_c.CentralDerivative_x (N, N, h, u, ux)
    IB_c.CentralDerivative_x (N, N, h, v, vx)
    IB_c.CentralDerivative_y (N, N, h, u, uy)
    IB_c.CentralDerivative_y (N, N, h, v, vy)

    Xss, Yss = FiberForce (Nb, hb, s, X, Y)
    fx, fy = ForceToGrid (N, N, h, Nb, hb, X, Y, Xss, Yss)
    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy, rho)
    #fx, fy = ExplicitTerms (dt, u, v, 0., 0., 0., 0., 0., 0., rho)
    
    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy, rho, mu)
    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, 1., X, Y, u, v)
    X += dt * Xvel
    Y += dt * Yvel


    if ShowStats:
        print "Explicit:", time.time() - Time

    Time = time.time() - Time
    TotalTime += Time
    count += 1

    if count % 50 == 0:
        P = ifft2(P).real
        clf()
        #imshow(u)
        imshow((u**2+v**2)**.5)
        #imshow(uy - vx)
        #colorbar()
        scatter(N*Y,N*X)



##    Ts = [.001,.002,.005,.01,.05]
##    for _T in Ts:
##        if CurT < _T and CurT + dt >= _T:
##            print "Time", _T, ":", TotalTime, TotalTime / count
##            Times.append(TotalTime)
    print "Time:", CurT, Time
    CurT += dt

print TotalTime
print TotalTime / float(count)
print dt, 1. / dt

Times = [TotalTime / count] + Times

def number_format(num, places=0):
    return locale.format("%.*f", (places, num), True)

for num in Times:
    print number_format(num, 3),'&',
    
