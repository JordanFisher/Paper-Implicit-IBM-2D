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

def OriginalMatrix (X, Y, _N, Nb, _h, hb, UField, VField, band = -1):
    if band == -1:
        band = Nb
        
    A = zeros((2*Nb,2*Nb),float32)
    B = zeros((2*Nb,2*Nb),float32)
    for i in range(Nb):
        for j in range(Nb):
            if abs(j - i) < band or abs(j - i) > Nb - band:
                a, b = (X[i]-X[j])/_h, (Y[i]-Y[j])/_h
                _i = floor(a)
                _j = floor(b)
                a, b = a - _i, b - _j

                if i >= j:
                    A[i,j] = (1.-a)*(1.-b) * hb * UField[(_i)%_N,(_j)%_N]
                    A[i+Nb,j+Nb] = (1.-a)*(1.-b) * hb * UField[(_j)%_N, (_i)%_N]

                    A[i,j] += (a)*(1.-b) * hb * UField[(_i+1)%_N,(_j)%_N]
                    A[i+Nb,j+Nb] += (a)*(1.-b) * hb * UField[(_j)%_N, (_i+1)%_N]
                
                    A[i,j] += (1.-a)*(b) * hb * UField[(_i)%_N,(_j+1)%_N]
                    A[i+Nb,j+Nb] += (1.-a)*(b) * hb * UField[(_j+1)%_N, (_i)%_N]

                    A[i,j] += (a)*(b) * hb * UField[(_i+1)%_N,(_j+1)%_N]
                    A[i+Nb,j+Nb] += (a)*(b) * hb * UField[(_j+1)%_N, (_i+1)%_N]

                    A[j,i] = A[i,j]
                    A[j+Nb,i+Nb] = A[i+Nb, j+Nb]

                A[i+Nb,j] = (1.-a)*(1.-b) * hb * VField[(_i)%_N,(_j)%_N]
                A[i+Nb,j] += (a)*(1.-b) * hb * VField[(_i+1)%_N,(_j)%_N]    
                A[i+Nb,j] += (1.-a)*(b) * hb * VField[(_i)%_N,(_j+1)%_N]
                A[i+Nb,j] += (a)*(b) * hb * VField[(_i+1)%_N,(_j+1)%_N]                            

                A[j,i+Nb] = A[i+Nb,j]


##                A[i,j] = (1.-a)*(1.-b) * hb * UField[(_i)%_N,(_j)%_N]
##                A[i+Nb,j] = (1.-a)*(1.-b) * hb * VField[(_i)%_N,(_j)%_N]
##                A[i,j+Nb] = (1.-a)*(1.-b) * hb * VField[(_j)%_N, (_i)%_N]
##                A[i+Nb,j+Nb] = (1.-a)*(1.-b) * hb * UField[(_j)%_N, (_i)%_N]
##
##                A[i,j] += (a)*(1.-b) * hb * UField[(_i+1)%_N,(_j)%_N]
##                A[i+Nb,j] += (a)*(1.-b) * hb * VField[(_i+1)%_N,(_j)%_N]
##                A[i,j+Nb] += (a)*(1.-b) * hb * VField[(_j)%_N, (_i+1)%_N]
##                A[i+Nb,j+Nb] += (a)*(1.-b) * hb * UField[(_j)%_N, (_i+1)%_N]
##
##                A[i,j] += (1.-a)*(b) * hb * UField[(_i)%_N,(_j+1)%_N]
##                A[i+Nb,j] += (1.-a)*(b) * hb * VField[(_i)%_N,(_j+1)%_N]
##                A[i,j+Nb] += (1.-a)*(b) * hb * VField[(_j+1)%_N, (_i)%_N]
##                A[i+Nb,j+Nb] += (1.-a)*(b) * hb * UField[(_j+1)%_N, (_i)%_N]
##
##                A[i,j] += (a)*(b) * hb * UField[(_i+1)%_N,(_j+1)%_N]
##                A[i+Nb,j] += (a)*(b) * hb * VField[(_i+1)%_N,(_j+1)%_N]
##                A[i,j+Nb] += (a)*(b) * hb * VField[(_j+1)%_N, (_i+1)%_N]
##                A[i+Nb,j+Nb] += (a)*(b) * hb * UField[(_j+1)%_N, (_i+1)%_N]


##    for i in range (2*Nb):
##        for j in range (2*Nb):
##            if abs(A[i,j] - B[i,j]) > .001:
##                print (i,j), A[i,j], B[i,j]

    return A


def FiberForce (Nb, hb, s, X, Y, Xss, Yss):
    SecondDerivativePeriodic (Nb, hb, X, Xss)
    SecondDerivativePeriodic (Nb, hb, Y, Yss)

    Xss, Yss = s * Xss, s * Yss

    return Xss, Yss

def ComputeDssMatrix (Nb, hb):
    DssMatrix = zeros((2*Nb,2*Nb), float32)
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



N = 2*128
_N = 2*128
Nb = 4*128
h = 1. / N
_h = 1. / _N
hb = 1. / Nb
dt = .0001
T = .005
CurT = 0
s = 100000.


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
##clf()
##scatter(X,Y,c='k',s=16)
##r = (.4*.2)**.5
##Nb /= 8
##Ellipse (Nb, X, Y, r, r, .5, .5)
##scatter(X,Y,c='k',s=10)




I = []
__N = Nb
while __N > 2:
    _I = zeros((2*__N,__N), float64)
    for i in range(__N/2):
        _I[2*i,i] = 1.
        _I[2*i+1,(i+1)%(__N/2)] = .5
        _I[2*i+1,(i)%(__N/2)] = .5

        _I[2*i+__N,i+__N/2] = 1.
        _I[2*i+1+__N,(i+1)%(__N/2)+__N/2] = .5
        _I[2*i+1+__N,(i)%(__N/2)+__N/2] = .5

    __N /= 2
    I.append(sparse.csc_matrix(_I))



A = zeros((2*Nb,2*Nb), float64)


Old_X, Old_Y = 1. * X, 1. * Y

b = zeros(2*Nb,float64)
count = 0
while CurT < T:
    print "Time:", CurT
    CurT += dt

    Old_u, Old_v = 1. * u, 1. * v
    Old_X, Old_Y = 1. * X, 1. * Y

    IB_c.CentralDerivative_x (N, N, h, u, ux)
    IB_c.CentralDerivative_x (N, N, h, v, vx)
    IB_c.CentralDerivative_y (N, N, h, u, uy)
    IB_c.CentralDerivative_y (N, N, h, v, vy)

#    Xss, Yss = FiberForce (Nb, hb, s, X, Y, Xss, Yss)
#    fx, fy = ForceToGrid (N, N, h, Nb, hb, Old_X, Old_Y, Xss, Yss)
#    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)
    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, 0., 0.)
    
    
    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, 1., X, Y, u, v)





    A = zeros((2*Nb,2*Nb),float64)
    IB_c.ComputeMatrix (X, Y, _N, _N, Nb, _h, hb, UField, VField, UField2, VField2, A, -1)#, band = Nb/8+1)

##    O = OriginalMatrix (X, Y, _N, Nb, _h, hb, UField, UField2, band = -1)
##    clf(); imshow(A-O); raw_input("")
##    raise
    
    P = identity(2*Nb,dtype=float64) - dt * dot(A,s*DssMatrix)
    clf(); imshow(A); raw_input("")    
    __P = [P]
    for i in range(len(I)):
        print i
        __P.append( array((I[i].T * matrix(__P[i]) * I[i]).todense()) )
  
    for i in range(Nb):
        b[i] = X[i] + dt * Xvel[i]
        b[i+Nb] = Y[i] + dt * Yvel[i]

    y = zeros(2*Nb,float64)
    y = MultigridStep(__P,b,I,y,.15)
    print "Res:", max(list(abs(dot(P,y)-b)))    
##    clf(); plot(y); raw_input("")
    y = MultigridStep(__P,b,I,y,.15)
    print "Res:", max(list(abs(dot(P,y)-b)))
#    y = linalg.solve(P,b)
    clf(); plot(y); raw_input("")



#    X, Y = X + y[0:Nb], Y + y[Nb:2*Nb]
    X, Y = y[0:Nb], y[Nb:2*Nb]
    scatter(X,Y,c='r'); raw_input("")




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
    print X - Old_X - dt*Xvel
    print Y - Old_Y - dt*Yvel    

    count += 1
    if count % 1 == 0:
        P = ifft2(P).real
        clf()
        imshow(P)
        scatter(N*Y,N*X)
        quiver(N*Y,N*X,Y-Old_Y,-(X-Old_X))
    raw_input("")
