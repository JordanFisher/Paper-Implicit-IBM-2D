from scipy.linalg import eig as Eig
import scipy
from pylab import *
from numpy import *
from IB_Methods import *
import IB_c
from scipy import sparse, linsolve
import math
from ValveSetup import PlotValve
#show()

def CompS (M, X, Y, N, h, Nb, hb):
    S = zeros((N**2, Nb))
    for i in range(Nb):
        u, v = 0. * u, 0. * v

        fx, fy = 0. * X, 0. * Y
        if i < Nb:
            fx[i] = 1.
        else:
            fy[i-Nb] = 1.
        fx, fy = ForceToGrid (N, N, h, Nb, hb, X, Y, fx, fy)

#        print "    ", fx[0,0], fx[0,1]
        fx, fy = ExplicitTerms (dt, u, v, 0., 0., 0., 0., fx, fy)
        
        fx = fft2(fx)
        fy = fft2(fy)

        P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
        P[0,0] = 0.   

        u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
        u = ifft2(u).real
        v = ifft2(v).real

        M[i,:Nb], M[i,Nb:] = VelToFiber (N, N, h, Nb, hb, dt, X, Y, u, v)
    

def CompMExact (M, X, Y, u, v, fx, fy, N, h, Nb, hb, dt):
    for i in range(2*Nb):
        u, v = 0. * u, 0. * v

        fx, fy = 0. * X, 0. * Y
        if i < Nb:
            fx[i] = 1.
        else:
            fy[i-Nb] = 1.
        fx, fy = ForceToGrid (N, N, h, Nb, hb, X, Y, fx, fy)
#        print "    ", fx[0,0], fx[0,1]
        fx, fy = ExplicitTerms (dt, u, v, 0., 0., 0., 0., fx, fy)
        
        fx = fft2(fx)
        fy = fft2(fy)

        P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
        P[0,0] = 0.   

        u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
        u = ifft2(u).real
        v = ifft2(v).real

        M[i,:Nb], M[i,Nb:] = VelToFiber (N, N, h, Nb, hb, dt, X, Y, u, v)


def GetEig(D, e1, e2, w):
    w -= dot(w,e1) * e1 / dot(e1,e1)
    w -= dot(w,e2) * e2 / dot(e2,e2)

    for i in range(3):
        w = linalg.solve(D,w)
        
        w -= dot(w,e1) * e1 / dot(e1,e1)
        w -= dot(w,e2) * e2 / dot(e2,e2)
        w /= dot(w,w)**.5        

    return w

def GetEigs(B):
    eigs, vecs = Eig(B)
    args = argsort(array(real(eigs)))
    return eigs[args], vecs[:,args]#vecs[args]

    
def Jacobian(z, Nb, Links, Tethers):
    D = zeros((2*Nb,2*Nb),float64)
    for k in range(2*Nb):
        _dF = dF(k, Nb, z[0:Nb], z[Nb:2*Nb], Links, Tethers)
        D[k,:] = _dF
    D = sparse.csc_matrix(D)
    return D

def dF(i, Nb, X, Y, Links, Tethers):
    _i = i % Nb
    
    _dA = zeros(2*Nb,float64)
    for link in Links:
        a, b, s, L = link
        if a == _i or b == _i:
            l = ((X[b] - X[a])**2 + (Y[b] - Y[a])**2)**.5

            if a == _i: j = b
            else: j = a

            if i < Nb:
                d = s * (-1. + L * (1./l - (X[b] - X[a])**2 / l**3))
                _dA[i] += d
                _dA[j] -= d

                d = -s * L * (X[j] - X[i]) * (Y[j] - Y[i]) / l**3
                _dA[i+Nb] += d
                _dA[j+Nb] -= d
            else:
                d = s * (-1. + L * (1./l - (Y[b] - Y[a])**2 / l**3))
                _dA[i] += d
                _dA[j+Nb] -= d

                d = -s * L * (X[j] - X[_i]) * (Y[j] - Y[_i]) / l**3
                _dA[_i] += d
                _dA[j] -= d


    for tether in Tethers:
        a, _x, s = tether
        if a == _i:
            _dA[i] += -s

    return _dA


N = 1*64
h = 1. / N
dt = .01*h#.00001
Nb = N
hb = 1. / Nb
s = 1000000. / hb**2

##Nb = int(.5 * N)
hb = 1. / Nb
x = zeros(2*Nb, float64)

UField, VField, UField2, VField2 = InitVelField(N, N, h, h, dt)
print abs(UField).max(), abs(VField).max()
##print UField[0,0], UField[4,0], UField[0,0] - UField[4,0]
##raise
A = zeros((2*Nb,2*Nb),float64)

u = zeros((N,N), float64)
v = zeros((N,N), float64)
fx = zeros((N,N), float64)
fy = zeros((N,N), float64)

WideLambda = zeros((N,N),float64)
ShortLambda = zeros((N,N),float64)
IB_c.InitWideLaplacian(N, N, h, WideLambda)
IB_c.InitShortLaplacian(N, N, h, ShortLambda)
DxSymbol = InitDxSymbol(N, N, h)
DySymbol = InitDySymbol(N, N, h)




########fx, fy = ForceToGrid (N, N, h, 1, 1, array([0],float64), array([0],float64), array([1],float64), array([0],float64))
########print fx[:,0]
########print sum(fx)
########
########fx = fft2(fx)
########fy = fft2(fy)
########
########P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
########P[0,0] = 0.   
########
########u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
########u = ifft2(u).real
########v = ifft2(v).real
########
########
########
########
########fx, fy = ForceToGrid (N, N, h, 1, 1, array([.25*h],float64), array([.25*h],float64), array([1],float64), array([0],float64))
########print sum(fx)
########print fx[:,0]
########
########fx = fft2(fx)
########fy = fft2(fy)
########
########P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
########P[0,0] = 0.   
########
########u2, v2 = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
########u2 = ifft2(u).real
########v2 = ifft2(v).real
########
########
########
########
########raise




e1 = 0. * x
e2 = 0. * x
for i in range(Nb):
    e1[i] = 1
    e2[i+Nb] = 1

data = []
for i in range(2*Nb):
    data.append([])
lGrid = []
l = 2. * 3.141592653 / Nb
while True:#l <= 2.1:
    X, Y, Links = [], [], []
##    Nb /= 2
    for i in range(Nb):
        X.append(.5 + .05*cos(2. * 3.141592653 / Nb * i))
        Y.append(.5 + .05*sin(2. * 3.141592653 / Nb * i))
##    r = .29875
##    for i in range(Nb):
##        X.append(.5 + r*cos(2. * 3.141592653 / Nb * i))
##        Y.append(.5 + r*sin(2. * 3.141592653 / Nb * i))
##    Nb *= 2

    fx = 0. * u
    fy = 0. * u
    Fx = zeros(Nb,float64)
    Fy = zeros(Nb,float64)

##  Square
##    for i in range(Nb/4):
##        X.append(.3 + .4 * i / (Nb/4))
##        Y.append(.3)
##        Fy[i] = 1.
##        
##        
##    for i in range(Nb/4):
##        Y.append(.3 + .4 * i / (Nb/4))
##        X.append(.7)
##        Fx[i+Nb/4] = -1.
##        
##    for i in range(Nb/4):
##        X.append(.7 - .4 * i / (Nb/4))
##        Y.append(.7)
##        Fy[i+Nb/2] = -1.
##        
##    for i in range(Nb/4):
##        Y.append(.7 - .4 * i / (Nb/4))
##        X.append(.3)
##        Fx[i+Nb/2+Nb/4] = 1.

##  Line
##    for i in range(Nb):
##        #X.append(.25 + .5 * i / (Nb))
##        X.append(10.1 * h + i * 6 * h / Nb)
##        Y.append(.5)
##        Fy[i] = 1.

##    X = array(X, float64)
##    Y = array(Y, float64)
##
##    fx, fy = ForceToGrid (N, N, h, Nb, hb, X, Y, Fx, Fy)
##    _fx, _fy = 1. * fx, 1. * fy
##
##    fx = fft2(fx)
##    fy = fft2(fy)
##
##    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
##    P[0,0] = 0.   
##
##    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
##    u = ifft2(u).real
##    v = ifft2(v).real
##
##    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, dt, X, Y, u, v)
##    P = ifft2(P).real
##    Px = 0. * P
##    Py = 0. * P
##    IB_c.CentralDerivative_x (N, N, h, P, Px)
##    IB_c.CentralDerivative_y (N, N, h, P, Py)    
##    raise

##    for i in range(Nb):
##        l = ((X[i]-X[(i+1)%Nb])**2+(Y[i]-Y[(i+1)%Nb])**2)**.5
##        #l = 0
##        Links.append((i,(i+1)%Nb,1.,l))

    x[:Nb], x[Nb:] = X, Y

    IB_c.ComputeMatrix (x[:Nb], x[Nb:], N, N, Nb, h, hb, UField, VField, UField2, VField2, A, -1)#N/6)
    A *= dt
    eigsA, vecsA = GetEigs(A)

    A2 = 1. * A
    CompMExact (A2, x[:Nb], x[Nb:], u, v, fx, fy, N, h, Nb, hb, dt)
    eigsA2, vecsA2 = GetEigs(A2)

    print A2[0,0] - A[0,0]

    X = array(X, float64)
    Y = array(Y, float64)
    for i in range(2*Nb):
        v = vecsA2[:,i].real
        fx, fy = ForceToGrid (N, N, h, Nb, hb, X, Y, v[:Nb], v[Nb:])
        print eigsA2[i], abs(fx).max(), abs(fy).max()

        fx = fft2(fx)
        fy = fft2(fy)

        P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
        P[0,0] = 0.   

        __u, __v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
        __u = ifft2(__u).real
        __v = ifft2(__v).real
        print "     ", abs(__u).max(), abs(__v).max()


        Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, dt, X, Y, __u, __v)        
        print "     ", abs(Xvel).max(), abs(Yvel).max()

        y = dot(A2, v) / dt
        print "     ", abs(y[:Nb]).max(), abs(y[Nb:]).max()
        
    
    raise

##    X = [1., 0., -1., 0.]
##    Y = [0., 1., 0., -1.]
##    Links = [(0,1,1.,l),(1,2,1.,l),(2,3,1.,l),(3,0,1.,l)]#,(0,2,1.,1.),(1,3,1.,1.)]

##    Nb = 3
##    X, Y = [], []
##    for i in range(3):
##        X.append(cos(2./3.*math.pi*i))
##        Y.append(sin(2./3.*math.pi*i))        
##    Links = [(0,1,1.,l),(1,2,1.,l),(2,0,1.,l)]





    J = s * Jacobian(x,Nb,Links,[]).todense()
    eigs, vecs = GetEigs(J)

    eigsA, vecsA = GetEigs(A)
    eigsA2, vecsA2 = GetEigs(A2)    

    eigsAJ, vecsAJ = GetEigs(dot(A,J))

    eigsP, vecsP = GetEigs(identity(2*Nb)-dot(A,J))

#    eigsS, vecsS = GetEigs(identity(2*Nb)/A[0,0]-J)

    eigsS, vecsS = GetEigs(dot(A, identity(2*Nb)/A[0,0]-J))

    for i in range(2*Nb):
        vecs[:,i] -= e1 * dot(e1, vecs[:,i]) / dot(e1, e1)
        vecs[:,i] -= e2 * dot(e2, vecs[:,i]) / dot(e2, e2)        

##    for i in range(2*Nb):
##        clf(); scatter(X,Y);quiver(X,Y,vecs[:Nb,i],vecs[Nb:,i])
##        print eigs[i]
##        raw_input("")

    raise

    eigs.sort()
    for i in range(2*Nb):
        data[i].append(float(real(eigs[i])))
    lGrid.append(l)
    print eigs

    l += .01

clf()
for i in range(2*Nb):
    plot(lGrid,data[i],c='k',lw=1.6)
axis([1.,2.,-3.,1.])

xlabel('Resting length')
ylabel('Eigenvalue magnitude')
