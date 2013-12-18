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


N = 16*32
_N = N
Nb = 2*N

h = 1. / N
_h = 1. / _N
hb = 1. / Nb
dt = .01*h#.1*h#.02 * h
#dt = .00003 * h
T = .005
CurT = 0
s = 100000.
#s = 10000000.

#Nb = 2

print "h:",h, ", dt:", dt, ", hb:", hb


WideLambda = zeros((N,N),float64)
ShortLambda = zeros((N,N),float64)
IB_c.InitWideLaplacian(N, N, h, WideLambda)
IB_c.InitShortLaplacian(N, N, h, ShortLambda)
DxSymbol = InitDxSymbol(N, N, h)
DySymbol = InitDySymbol(N, N, h)

print abs(1. / (1. - dt * WideLambda)).sum()

##_h = h
##_N, _M = int(1. / _h), int(1. / _h)
##UField, VField, UField2, VField2 = InitVelField(_N, _M, _h, h, dt)
##
##print UField[0,0],VField2[0,0]


u = zeros((N,N), float64)
fx = 0. * u
fx[0,0] = 1.
fx = fft2(fx)

P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, 0.)
P[0,0] = 0.   

u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, 0)
u = ifft2(u).real
v = ifft2(v).real

print u.max() / h**2



########################raise
########################clf()
########################clf(); imshow(UField); colorbar(); #show(); #&&raw_input("")
########################clf(); imshow(VField2); colorbar(); #show(); #&&raw_input("")
########################
#########################Nb = 2
########################
########################X = zeros(Nb, float64)
########################Y = zeros(Nb, float64)
########################Xss = zeros(Nb, float64)
########################Yss = zeros(Nb, float64)
########################u = zeros((N,N), float64)
########################v = zeros((N,N), float64)
########################ux = zeros((N,N), float64)
########################vx = zeros((N,N), float64)
########################uy = zeros((N,N), float64)
########################vy = zeros((N,N), float64)
########################fx = zeros((N,N), float64)
########################fy = zeros((N,N), float64)
########################
########################Ellipse (Nb, X, Y, .4, .2, .5, .5)
##########################X[0] = 0.234 * h
##########################Y[0] = 0.5643 * h
##########################X[1] = X[0] + h
##########################Y[1] = Y[0] + h
########################
##########################X[0] = 0.
##########################Y[0] = 0.
##########################X[1] = 1.
##########################Y[0] = 1.
########################
########################
########################
########################def FiberForce (Nb, hb, s, X, Y, Xss, Yss):
########################    SecondDerivativePeriodic (Nb, hb, X, Xss)
########################    SecondDerivativePeriodic (Nb, hb, Y, Yss)
########################
########################    Xss, Yss = s * Xss, s * Yss
########################
########################    return Xss, Yss
########################
########################
########################def CompMExact (M, X, Y, u, v, fx, fy, N, h, Nb, hb, dt):
########################    for i in range(2*Nb):
########################        u, v = 0. * u, 0. * v
########################
########################        fx, fy = 0. * X, 0. * Y
########################        if i < Nb:
########################            fx[i] = 1.
########################        else:
########################            fy[i-Nb] = 1.
########################        fx, fy = ForceToGrid (N, N, h, Nb, hb, X, Y, fx, fy)
#########################        print "    ", fx[0,0], fx[0,1]
########################        fx, fy = ExplicitTerms (dt, u, v, 0., 0., 0., 0., fx, fy)
########################        
########################        fx = fft2(fx)
########################        fy = fft2(fy)
########################
########################        P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
########################        P[0,0] = 0.   
########################
########################        u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
########################        u = ifft2(u).real
########################        v = ifft2(v).real
########################
########################        M[i,:Nb], M[i,Nb:] = VelToFiber (N, N, h, Nb, hb, dt, X, Y, u, v)
########################
########################
########################
########################
########################A1 = zeros((2*Nb,2*Nb),float64)
########################IB_c.ComputeMatrix (X, Y, _N, _N, Nb, _h, hb, UField, VField, UField2, VField2, A1, -1)
########################A1 *= dt
########################
########################A2 = zeros((2*Nb,2*Nb),float64)
########################CompMExact (A2, X, Y, u, v, fx, fy, N, h, Nb, hb, dt)
########################
########################P = A1 - A2
##########################print P[0,0]#/A2[0,0]
##########################print A2[0,0]
##########################print A1[0,0]
########################
########################r1 = range(1,Nb-1)
########################r2 = range(Nb-2)
########################print max(abs(P[r1,r2])),
########################
########################f = zeros(2*Nb, float64)
########################f[:Nb], f[Nb:] = FiberForce (Nb, hb, s, X, Y, Xss, Yss)
########################a = dot(A1, f)
########################b = dot(A2, f)
########################print max(abs(a-b))
