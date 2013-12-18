import sys
sys.path.append("c:\HeartValve")
sys.path.append("c:\HeartValve\IB_c\Release")
from scipy.linalg import eig as Eig
import scipy
from pylab import *
from numpy import *
from numpy.fft import fft2, ifft2, fft as FFT, ifft as IFFT, rfft
from IB_Methods import *
import IB_c
from ValveSetup import *
import time
import math
show()

def FluidSolve (u, v, ux, vx, uy, vy, X, Y, Fx, Fy):
    IB_c.CentralDerivative_x (N, M, h, u, ux)
    IB_c.CentralDerivative_x (N, M, h, v, vx)
    IB_c.CentralDerivative_y (N, M, h, u, uy)
    IB_c.CentralDerivative_y (N, M, h, v, vy)

    fx, fy = ForceToGrid (N, M, h, Nb, hb, Old_X, Old_Y, Fx, Fy)
    
    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)
#    fx += dt * Current * ones((N,M),float64)

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

print UField[0,0]
raise
U = real(fft2(UField))
U[:10,:10] *= 0
U[N-10:,:10] *= 0
U[:10,N-10:] *= 0
U[N-10:,N-10:] *= 0

# Quarter plane
F = zeros((N**2/4,N**2/4))
for j1 in range(N/2):
    for k1 in range(N/2):
        for j2 in range(N/2):
            for k2 in range(N/2):
                #F[j1 + N/2 * k1, j2 + N/2 * k2] = VField[abs(j1 - j2), abs(k1 - k2)]
                F[j1 + N/2 * k1, j2 + N/2 * k2] = U[abs(j1 - j2), abs(k1 - k2)]                


### N/8 < x < 3N/8, y < N/8 or y > 3N/8
##x_width = N/8
##x_start = (N/2 - x_width) / 2
##mult = 1
##y_width = (N/2 - mult*x_width) / 2
##F = zeros(((x_width)**2,(2*y_width)**2))
##for j1 in range(x_width):
##    for k1 in range(x_width):
##        for j2 in range(2*y_width):
##            for k2 in range(2*y_width):
##                _j2, _k2 = j2, k2
##                if (j2 >= y_width): _j2 += mult*x_width
##                if (k2 >= y_width): _k2 += mult*x_width          
##                F[j1 + x_width * k1, j2 + 2*y_width * k2] = UField[abs(j1+x_start - _j2), abs(k1+x_start - _k2)]



### x on BL, y on TR quadrant
##F = zeros(((N/4)**2,(N/4)**2))
##for j1 in range(N/4):
##    for k1 in range(N/4):
##        for j2 in range(N/4):
##            for k2 in range(N/4):
##                #F[j1 + N/4 * k1, j2 + N/4 * k2] = VField[abs(j1 - j2 - N/4), abs(k1 - k2 - N/4)]
##                F[j1 + N/4 * k1, j2 + N/4 * k2] = U[abs(j1 - j2 - N/4), abs(k1 - k2 - N/4)]

### x on BL, y on TR quadrant, plus small overlap
##F = zeros(((N/4)**2,(N/4)**2))
##for j1 in range(N/4):
##    for k1 in range(N/4):
##        for j2 in range(N/4):
##            for k2 in range(N/4):
##                #F[j1 + N/4 * k1, j2 + N/4 * k2] = VField[abs(j1 - j2 - N/4), abs(k1 - k2 - N/4)]
##                F[j1 + N/4 * k1, j2 + N/4 * k2] = U[abs(j1 - (j2 + N/4 - 2)), abs(k1 - (k2 + N/4 - 2))]


### x on left, y on right
##F = zeros((N**2/8,N**2/8))
##for j1 in range(N/4):
##    for k1 in range(N/2):
##        for j2 in range(N/4):
##            for k2 in range(N/2):
##                F[j1 + N/4 * k1, j2 + N/4 * k2] = UField[abs(j1 - j2 - N/4), abs(k1 - k2)]




# Predict for all y > x (both components)
def Predict(F, w):
    for j1 in range(N/2):
        for k1 in range(N/2):
            for j2 in range(j1,N/2):
                for k2 in range(k1,N/2):
                    F[j1 + N/2 * k1, j2 + N/2 * k2] = w[j1 + N/2 * k1, j2 + N/2 * k2]

# Predict for all abs(x - y) < 5 in either component
a = 5
def Predict(F, w):
    for j1 in range(N/2):
        for k1 in range(N/2):
            for j2 in range(max(0,j1-5),min(N/2,j1+5)):
                for k2 in range(max(0,k1-5),min(N/2,k1+5)):
                    F[j1 + N/2 * k1, j2 + N/2 * k2] = w[j1 + N/2 * k1, j2 + N/2 * k2]

t = 20
def Iter():
    U,s,V = svd(F)    
    w = dot(U[:,:t], dot(diag(s[:t]), V[:t,:]))
    print dot(s[t:],s[t:])**.5, abs(F-w).max()
    Predict(F,w)

def Get(t):
    U,s,V = svd(F)    
    w = dot(U[:,:t], dot(diag(s[:t]), V[:t,:]))
    print dot(s[t:],s[t:])**.5, abs(F-w).max()
    return U[:,:t], V[:t,:]
