import sys
sys.path.append("c:\HeartValve")
sys.path.append("c:\HeartValve\IB_c\Release")
from scipy.linalg import eig as Eig
from pylab import *
from numpy import *
from numpy.fft import fft2, ifft2, fft as FFT, ifft as IFFT, rfft
from IB_Methods import *
import IB_c
from ValveSetup import *
from scipy import sparse, linsolve
import time
import math
#show()


def GetEig(D, e1, e2, w):
    w -= dot(w,e1) * e1 / dot(e1,e1)
    w -= dot(w,e2) * e2 / dot(e2,e2)

    for i in range(3):
        w = linalg.solve(D,w)
        
        w -= dot(w,e1) * e1 / dot(e1,e1)
        w -= dot(w,e2) * e2 / dot(e2,e2)
        w /= dot(w,w)**.5        

    return w

def GetEigSparse(D, e1, e2, w):
    w -= dot(w,e1) * e1 / dot(e1,e1)
    w -= dot(w,e2) * e2 / dot(e2,e2)

    for i in range(3):
        w = linsolve.spsolve(D,w)
        
        w -= dot(w,e1) * e1 / dot(e1,e1)
        w -= dot(w,e2) * e2 / dot(e2,e2)
        w /= dot(w,w)**.5        

    return w


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
#        Xss[a] += -s*X[a]
#        Yss[a] += -s*Y[a]

    return Xss, Yss
    
def Jacobian(z, Nb, Links, Tethers):
    D = zeros((2*Nb,2*Nb),float64)
    for k in range(2*Nb):
        _dF = dF(k, Nb, z[0:Nb], z[Nb:2*Nb], Links, Tethers)
        D[k,:] = _dF
    D = sparse.csc_matrix(D)
    return D

def QuickJacobian(z, Nb, Links, Tethers, D):
    for link in Links:
        a, b, s, L = link
        
        l = ((z[b] - z[a])**2 + (z[b+Nb] - z[a+Nb])**2)**.5
        l3 = l**3
        l1 = 1./l

        dxdx = s * (-1. + L * (l1 - (z[b] - z[a])**2 / l3))
        dydy = s * (-1. + L * (l1 - (z[b+Nb] - z[a+Nb])**2 / l3))
        dxdy = -s * L * (z[b] - z[a]) * (z[b+Nb] - z[a+Nb]) / l3
        D[a,a] += dxdx
        D[a,b] -= dxdx
        D[b,a] -= dxdx
        D[b,b] += dxdx

        D[a+Nb,a+Nb] += dydy
        D[a+Nb,b+Nb] -= dydy
        D[b+Nb,a+Nb] -= dydy
        D[b+Nb,b+Nb] += dydy

        D[a,a+Nb] += dxdy
        D[a,b+Nb] -= dxdy
        D[b+Nb,a] -= dxdy
        D[a+Nb,a] += dxdy

        D[b,b+Nb] += dxdy
        D[b,a+Nb] -= dxdy
        D[a+Nb,b] -= dxdy
        D[b+Nb,b] += dxdy

    for tether in Tethers:
        a, _x, s = tether
        D[a,a] -= s
        D[a+Nb,a+Nb] -= s



def _FiberForce (Nb, x, Xss, Yss, Links, Tethers):
    Xss, Yss = FiberForce (Nb, x[:Nb], x[Nb:], Xss, Yss, Links, Tethers)
    f = 1. * x
    f[:Nb], f[Nb:] = 1.*Xss, 1.*Yss
    return f
  


Domain_x, Domain_y = 2., 1.
#h = .003
h = Domain_x / 100.
N, M = int(Domain_x / h), int(Domain_y / h)
Domain_x, Domain_y = N * h, M * h
print "N, M =", N, M
dt = .003
T = 5.
CurT = 0

_s = 5000000000.
Current = 100.

WideLambda = zeros((N,M),float64)
ShortLambda = zeros((N,M),float64)
IB_c.InitWideLaplacian(N, M, h, WideLambda)
IB_c.InitShortLaplacian(N, M, h, ShortLambda)
DxSymbol = InitDxSymbol(N, M, h)
DySymbol = InitDySymbol(N, M, h)

_h = h
_N, _M = int(Domain_x/ _h), int(Domain_y / _h)
UField, VField, UField2, VField2 = InitVelField(_N, _M, _h, h, dt)

print UField[0,0],VField2[0,0]
clf()
clf(); imshow(UField); colorbar(); #show(); #&&raw_input("")
clf(); imshow(VField2); colorbar(); #show(); #&&raw_input("")

X, Y, Links, Tethers, Parts = ConstructValve (_s, 0., Domain_x, 0., Domain_y, .015)#, True,1.,1.)#.01)


s2, __I, _x2, _y2, Links2, Tethers2, Parts2 = ReduceValve(_s, X, Y, Links, Tethers, Parts)
Nb2 = len(_x2)


_II = 1. * __I

_n, _m = _II.shape
__II = zeros((2*_n,2*_m),float64)
__II[:_n,:_m] = 1. * _II
__II[_n:,_m:] = 1. * _II
II = [sparse.csc_matrix(__II)]
s3, _x3, _y3, Links3, Tethers3, Parts3 = s2, _x2, _y2, Links2, Tethers2, Parts2
while len(_II) > 10 and len(II) < 7:
    holdlen = len(_II)
    print holdlen
    s3, _II, _x3, _y3, Links3, Tethers3, Parts3 = ReduceValve(s3, _x3, _y3, Links3, Tethers3, Parts3)
    if len(_II) == holdlen:
        II = II[:len(II)-2]
        break
    _n, _m = _II.shape
    __II = zeros((2*_n,2*_m),float64)
    __II[:_n,:_m] = 1. * _II
    __II[_n:,_m:] = 1. * _II
    II.append(sparse.csc_matrix(__II))


Nb = len(X)
print "Nb =", Nb
hb = 1. / Nb
hb = 1.


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

Xss2 = zeros(Nb2, float64)
Yss2 = zeros(Nb2, float64)


PlotValve (X, Y, Links)


A = zeros((2*Nb,2*Nb), float64)
x = zeros(2*Nb,float64)
_e1, _e2, _e3, F = 0*x, 0*x, 0*x, 0*x
e1, e2, e3 = 0. * x, 0. * x, 0. * x

Old_X, Old_Y = 1. * X, 1. * Y

b = zeros(2*Nb,float64)
count = 0
while CurT < T:
    print "Time:", CurT
    CurT += dt

    Predict_X, Predict_Y = X + .5 * (X - Old_X), Y + .25 * (Y - Old_Y)

    Old_u, Old_v = 1. * u, 1. * v
    Old_X, Old_Y = 1. * X, 1. * Y

    IB_c.CentralDerivative_x (N, M, h, u, ux)
    IB_c.CentralDerivative_x (N, M, h, v, vx)
    IB_c.CentralDerivative_y (N, M, h, u, uy)
    IB_c.CentralDerivative_y (N, M, h, v, vy)

    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, 0., 0.)
    fx += dt * Current * ones((N,M),float64)

    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, M, h, Nb, hb, 1., X, Y, u, v)





    A = zeros((2*Nb,2*Nb),float64)
    IB_c.ComputeMatrix (X, Y, _N, _M, Nb, _h, hb, UField, VField, UField2, VField2, A, -1)#, band = Nb/8+1)

##    print "Comping eigs"
##    eigs,vecs = Eig(A)
##    print eigs
##    raw_input("")


    __A = [dt*A]
    for i in range(len(II)):
        print i
#        __A.append( array(dot(transpose(II[i].todense()),dot(__A[i],II[i].todense()))) )
        __A.append( array((II[i].T * matrix(__A[i]) * II[i]).todense()) )
  
    for i in range(Nb):
        b[i] = X[i] + dt * Xvel[i]
        b[i+Nb] = Y[i] + dt * Yvel[i]


    x[0:Nb], x[Nb:2*Nb] = 1. * X, 1. * Y

    for p in range(2,7):
        for i in Parts[p]:
            e1[i] = 1.
            e2[i+Nb] = 1.

    D = zeros((2*Nb,2*Nb),float64)



    f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
    xp = b + dt*dot(A,f)

    for i in range(4):
        x = x + e1 * dot(xp-x,e1)/dot(e1,e1) + e2 * dot(xp-x,e2)/dot(e2,e2)

        center = e1*dot(x,e1)/dot(e1,e1) + e2*dot(x,e2)/dot(e2,e2) 

        z = x - center
        e3 = 0. * x
        for p in range(2,7):
            for i in Parts[p]:
                e3[i] = z[i+Nb]
                e3[i+Nb] = -z[i]
        theta = dot(xp-x,e3)/dot(e3,e3)

        Rot = zeros((2,2),float64)
        Rot[0,0] = cos(theta)
        Rot[1,0] = -sin(theta)
        Rot[0,1] = -Rot[1,0]
        Rot[1,1] = Rot[0,0]

        for p in range(2,7):
            for i in Parts[p]:
                [z[i],z[i+Nb]] = dot(Rot, [z[i],z[i+Nb]])

        x = z + center


    _e1 = MultigridStep(__A,e1,II,_e1,1.)
    _e2 = MultigridStep(__A,e2,II,_e2,1.)

    ChangeX = 1.   
    MaxRes = 1.
    count = 0
    F = 0. * F
    while count < 2:
        F = 0. * F
        count += 1
        holdx = 1. * x
        

        F = MultigridStep(__A,x-b,II,F,1.)
#        F = MultigridStep(__A,x-b,II,F,1.)
#        F = linalg.solve(dt*A, x-b)
        _f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
        f = F - _f

####        # Solve inverse translation eigenvectors
####        e3 = 0. * x
####        z = x - e1*dot(x,e1)/dot(e1,e1) - e2*dot(x,e2)/dot(e2,e2)
####        for p in range(2,7):
####            for i in Parts[p]:
####                e3[i] = z[i+Nb]
####                e3[i+Nb] = -z[i]
####
####
#####        _e3 = linalg.solve(dt*A,e3) - Dsp*e3
####        _e3 = MultigridStep(__A,e3,II,_e3,1.)
####        
####        K = zeros((3,3))
####        K[0,0] = dot(_e1,e1)
####        K[0,1] = dot(_e2,e1)
####        K[0,2] = dot(_e3,e1)
####        K[1,0] = dot(_e1,e2)
####        K[1,1] = dot(_e2,e2)
####        K[1,2] = dot(_e3,e2)
####        K[2,0] = dot(_e1,e3)
####        K[2,1] = dot(_e2,e3)
####        K[2,2] = dot(_e3,e3)
####
####        #_K = linalg.solve(K,array([dot(f,e1), dot(f,e2), dot(f,e3)]))
####        _K = linalg.solve(K,array([dot(F,e1), dot(F,e2), dot(F,e3)]))
####        if True:
####            x -= e1 * _K[0] + e2 * _K[1]# + e3 * _K[2]
####
####            center = e1*dot(x,e1)/dot(e1,e1) + e2*dot(x,e2)/dot(e2,e2) 
####            z = x - center
####            e3 = 0. * x
####
####            theta = -_K[2]
####            Rot = zeros((2,2),float64)
####            Rot[0,0] = cos(theta)
####            Rot[1,0] = -sin(theta)
####            Rot[0,1] = -Rot[1,0]
####            Rot[1,1] = Rot[0,0]
####
####            for p in range(2,7):
####                for i in Parts[p]:
####                    [z[i],z[i+Nb]] = dot(Rot, [z[i],z[i+Nb]])
####            x = z + center
####
####            print "  ", _K
####        else:
####            print "Skip this first step"



        maxf = max(list(abs(F)))
        I2 = zeros((2*Nb,2*Nb),float64)
        for i in range(2*Nb):
            if abs(F[i]) > 2:#.05 * maxf:
                I2[i,i] = 1.
        I1 = identity(2*Nb,dtype=float64) - I2

        clf(); plot(F); plot(dot(I2,F),c='r'); raw_input("")

        subMax = 100.                
        while subMax > .000001:
#            clf(); scatter(x[:Nb],x[Nb:]); raw_input("")
            
            D = 0. * D
            QuickJacobian(x, Nb, Links, Tethers, D)
            D = Jacobian(x, Nb, Links, Tethers).todense()
            Dsp = sparse.csc_matrix(array(D))

            _f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
            f = F - _f

            r = dot(I1,f) + dot(I2,b-x+dot(dt*A,_f))
            P = identity(2*Nb)-dt*dot(A,D)
            y = linalg.solve(dot(I1,D)+dot(I2,P),r)

            x += y
            subMax = max(list(abs(r)))
            print subMax

########
########        subMax = 1000.
########        damp = 1.
########        _holdx = 1. * x
########        holdrex = 0
########        while subMax > .00001:
#########            clf(); scatter(x[:Nb],x[Nb:]); raw_input("")
########            
########            D = 0. * D
########            QuickJacobian(x, Nb, Links, Tethers, D)
########            D = Jacobian(x, Nb, Links, Tethers).todense()
########            Dsp = sparse.csc_matrix(array(D))
########
########            _f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
########            f = F - _f
########
########            e3 = 0. * x
########            z = x - e1*dot(x,e1)/dot(e1,e1) - e2*dot(x,e2)/dot(e2,e2)
########            for p in range(2,7):
########                for i in Parts[p]:
########                    e3[i] = z[i+Nb]
########                    e3[i+Nb] = -z[i]
########
########            f -= e1 * dot(f,e1) / dot(e1,e1)
########            f -= e2 * dot(f,e2) / dot(e2,e2)
########            f -= e3 * dot(f,e3) / dot(e3,e3)
########
########            holdres = subMax
########            subMax = max(list(abs(f)))
########            print "       submax:",subMax, "  damp:",damp
########            if subMax > 10*holdres:
########                print "woah!"
########                x = 1. * _holdx
########                damp *= .5
########
########                e3 = 0. * x
########                z = x - e1*dot(x,e1)/dot(e1,e1) - e2*dot(x,e2)/dot(e2,e2)
########                for p in range(2,7):
########                    for i in Parts[p]:
########                        e3[i] = z[i+Nb]
########                        e3[i+Nb] = -z[i]
########                
########
########                D = 0. * D
########                QuickJacobian(x, Nb, Links, Tethers, D)
########                D = Jacobian(x, Nb, Links, Tethers).todense()
########                Dsp = sparse.csc_matrix(array(D))
########
########                _f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
########                f = F - _f
########
########                f -= e1 * dot(f,e1) / dot(e1,e1)
########                f -= e2 * dot(f,e2) / dot(e2,e2)
########                f -= e3 * dot(f,e3) / dot(e3,e3)
########
########
########           
########            y = linsolve.spsolve(Dsp,f)
########            y -= e1 * dot(y,e1) / dot(e1,e1)
########            y -= e2 * dot(y,e2) / dot(e2,e2)
########            y -= e3 * dot(y,e3) / dot(e3,e3)
########
########            _holdx = 1. * x
########            x += damp * y

        f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
        r = b + dot(dt*A,f) - x

        MaxRes = max(list(abs(r)))
        print "Residual:", MaxRes

        ChangeX = max(list(abs(x-holdx)))
        print "Change:", ChangeX


    f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
    r = b + dot(dt*A,f) - x

    print "Residual:", max(list(abs(r)))
    print
    print
        



    X, Y = 1.*x[0:Nb], 1.*x[Nb:2*Nb]



    u, v = 1. * Old_u, 1. * Old_v
    IB_c.CentralDerivative_x (N, M, h, u, ux)
    IB_c.CentralDerivative_x (N, M, h, v, vx)
    IB_c.CentralDerivative_y (N, M, h, u, uy)
    IB_c.CentralDerivative_y (N, M, h, v, vy)

    Xss, Yss = FiberForce (Nb, X, Y, Xss, Yss, Links, Tethers)   

    f = linalg.solve(dt*A,x-b)
    Xss, Yss = 1.*f[:Nb], 1.*f[Nb:]

    fx, fy = ForceToGrid (N, M, h, Nb, hb, Old_X, Old_Y, Xss, Yss)
    
    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)
    fx += dt * Current * ones((N,M),float64)

    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)

    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, M, h, Nb, hb, 1., Old_X, Old_Y, u, v)
    _X, _Y = Old_X + dt*Xvel, Old_Y + dt*Yvel
   
    count += 1
    if count % 1 == 0:
        P = ifft2(P).real
        clf()
        imshow(P)
        scatter(M*Y,N*X/2)
        scatter(_X,_Y,c='r')
        quiver(M*Y,N*X/2,Y-Old_Y,-(X-Old_X))
        #show()
    #&&raw_input("")
