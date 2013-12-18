import sys
import os
sys.path.append("c:\HeartValve")
sys.path.append("c:\HeartValve\IB_c\Release")
from scipy.linalg import eig as Eig
import scipy
from pylab import *
from numpy import *
import numpy
from numpy.fft import fft2, ifft2, fft as FFT, ifft as IFFT, rfft
from IB_Methods import *
import IB_c
from ValveSetup import *
import time
import math
#show()

def Shift(f, x, y):
    w = f * 0
    w[x:, y:] = f[:-x, :-y]
    w[x:, :y] = f[:-x, -y:]
    w[:x, y:] = f[-x:, :-y]
    w[:x, :y] = f[-x:, -y:]
    return w

def FluidSolve (u, v, ux, vx, uy, vy, X, Y, Fx, Fy):
    IB_c.CentralDerivative_x (N, M, h, u, ux)
    IB_c.CentralDerivative_x (N, M, h, v, vx)
    IB_c.CentralDerivative_y (N, M, h, u, uy)
    IB_c.CentralDerivative_y (N, M, h, v, vy)

    fx, fy = ForceToGrid (N, M, h, Nb, hb, Old_X, Old_Y, Fx, Fy)
    
    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)

    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)

    u = ifft2(u).real
    v = ifft2(v).real

    return VelToFiber (N, M, h, Nb, hb, 1., X, Y, u, v)


  
N = 64

Domain_x, Domain_y = 1., 1.
h = Domain_x / N
N, M = int(Domain_x / h), int(Domain_y / h)
dt = .002

WideLambda = zeros((N,M),float64)
ShortLambda = zeros((N,M),float64)
IB_c.InitWideLaplacian(N, M, h, WideLambda)
IB_c.InitShortLaplacian(N, M, h, ShortLambda)
DxSymbol = InitDxSymbol(N, M, h)
DySymbol = InitDySymbol(N, M, h)

_h = h
_N, _M = int(Domain_x/ _h), int(Domain_y / _h)
UField, VField, UField2, VField2 = InitVelField(_N, _M, _h, h, dt)

##UField = ones((N, N), float64)

##UField = zeros((N,N), float64)
##for j in range(N):
##    for k in range(N):
##        UField[j,k] = cos(2*j*pi*h)


print UField[0,0]



def Eval(j, U, V):
    result = 0
    for i in range(len(U)):
        result += U[i][N/2,N/2] * V[i][N/2 + j[0],N/2 + j[1]]

    return result

def EvalDirect(j, UField):
    return UField[j[0], j[1]]




##PhiHat = fft2(UField)
PhiHat = fft2(UField)
def Convolve(x):
    xhat = fft2(x)
    return real(ifft2(PhiHat * xhat))

    

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.isdir(d):
        os.makedirs(d)
        
root = 'c:\HeartValve2\\Decomposition\\'
##raise

Terms = 20

U_width = N / 4
U_width /= 2
while U_width > 4:
    U, V = [], []
    for l in range(Terms):
        U.append(ones((N,N)))
        V.append(ones((N,N)))
        for j in range(N):
            for k in range(N):
                V[l][j,k] = cos(j) * sin(k)

    # x in [U_x1,U_x2]x[U_y1,U_y2], y outside of [V_x1,V_x2]x[V_y1,V_y2]
    V_width = 3 * U_width

    U_x1 = N / 2 - U_width / 2
    U_x2 = N / 2 + U_width / 2
    U_y1 = N / 2 - U_width / 2
    U_y2 = N / 2 + U_width / 2
    U_x1 -= 1
    U_x2 += 1
    U_y1 -= 1
    U_y2 += 1

    V_x1 = N / 2 - V_width / 2
    V_x2 = N / 2 + V_width / 2
    V_y1 = N / 2 - V_width / 2
    V_y2 = N / 2 + V_width / 2
    V_x1 += 1
    V_x2 -= 1
    V_y1 += 1
    V_y2 -= 1

    def Enforce_U_Restriction(U):
        hold = U[U_x1:U_x2+1, U_y1:U_y2+1].copy()
        U *= 0
        U[U_x1:U_x2 + 1,U_y1:U_y2 + 1] = hold
        return U

    def Enforce_V_Restriction(V):
        V[V_x1:V_x2 + 1,V_y1:V_y2 + 1] *= 0
        return V

    _s = []
    for l in range(Terms):
        V[l] = Enforce_V_Restriction(V[l])
        
        for i in range(30):
            less = 0
            for _l in range(l):
                less += U[_l]*(V[_l]*V[l]).sum()
            U[l] = (Convolve(V[l]) - less) / (V[l]**2).sum()
            U[l] = Enforce_U_Restriction(U[l])

            # Renormalize
            norm1 = (U[l]**2).sum()**.5
            norm2 = (V[l]**2).sum()**.5
            U[l] *= (norm2 / norm1)**.5
            V[l] *= (norm1 / norm2)**.5

            less = 0
            for _l in range(l):
                less += U[_l]*(V[_l]*U[l]).sum()
            V[l] = (Convolve(U[l]) - less) / (U[l]**2).sum()
            V[l] = Enforce_V_Restriction(V[l])

            # Renormalize
            norm1 = (U[l]**2).sum()**.5
            norm2 = (V[l]**2).sum()**.5
            U[l] *= (norm2 / norm1)**.5
            V[l] *= (norm1 / norm2)**.5        
        print l,
            
        _s.append( ((U[l]**2).sum() * (V[l]**2).sum())**.5 )
    _s.sort()
    _s.reverse()

    raise

    for l in range(Terms):
        name = root + 'N_' + str(N) + '\\UField\\Panel_Width_' + str(U_width) + '\\U\\' +  str(l) + '.npy'
        ensure_dir(name)
        numpy.save(name, Shift(U[l], N/2, N/2))

        name = root + 'N_' + str(N) + '\\UField\\Panel_Width_' + str(U_width) + '\\V\\' +  str(l) + '.npy'
        ensure_dir(name)
        numpy.save(name, Shift(V[l], N/2, N/2))
        
    U_width /= 2
