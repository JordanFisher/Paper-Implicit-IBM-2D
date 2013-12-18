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
from scipy import sparse
import time
import random
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
 


def _FiberForce (Nb, x, Xss, Yss, Links, Tethers):
    Xss, Yss = FiberForce (Nb, x[:Nb], x[Nb:], Xss, Yss, Links, Tethers)
    f = 1. * x
    f[:Nb], f[Nb:] = 1.*Xss, 1.*Yss
    return f
  


Domain_x, Domain_y = 2., 1.
h = .003
#h =.0015
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
clf(); imshow(UField); colorbar(); #show(); raw_input("")
clf(); imshow(VField2); colorbar(); #show(); raw_input("")

X, Y, Links, Tethers, Parts = ConstructValve (_s, 0., Domain_x, 0., Domain_y, .015, True,1.5,1.15)#.01)
s2, __I, _x2, _y2, Links2, Tethers2, Parts2 = ReduceValve(_s, X, Y, Links, Tethers, Parts)
Nb2 = len(_x2)


_II = 1. * __I

_n, _m = _II.shape
__II = zeros((2*_n,2*_m),float64)
__II[:_n,:_m] = 1. * _II
__II[_n:,_m:] = 1. * _II
II = [sparse.csc_matrix(__II)]
s3, _x3, _y3, Links3, Tethers3, Parts3 = s2, _x2, _y2, Links2, Tethers2, Parts2
while len(_II) > 10:
    s3, _II, _x3, _y3, Links3, Tethers3, Parts3 = ReduceValve(s3, _x3, _y3, Links3, Tethers3, Parts3)
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

    __A = [dt*A]
    for i in range(len(II)):
        print i
        __A.append( array(dot(transpose(II[i].todense()),dot(__A[i],II[i].todense()))) )

  
    for i in range(Nb):
        b[i] = X[i] + dt * Xvel[i]
        b[i+Nb] = Y[i] + dt * Yvel[i]


    x = zeros(2*Nb,float64)
    x[0:Nb], x[Nb:2*Nb] = 1. * X, 1. * Y
    e3 = 1. * x

    e1 = zeros(2*Nb,float64)
    e2 = zeros(2*Nb,float64)

    for p in range(2,7):
        for i in Parts[p]:
            e1[i] = 1.
            e2[i+Nb] = 1.



    MaxRes = 1.
    its = 0
    while MaxRes > .0001 and its < 1:
        print "its", its
        its += 1

        if its == 1:
            f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
            update_x = 1. * b + 1. * dot(dt*A,f)
            _x = 1. * update_x

            clf()
            scatter(x[:Nb],x[Nb:])
            raw_input("")
            scatter(_x[:Nb],_x[Nb:],c='r')
            raw_input("")

            a1 = dot(_x,e1) / dot(e1,e1)
            a2 = dot(_x,e2) / dot(e2,e2)
            _x -= a1 * e1 + a2 * e2
            __x = 1. * x
            _a1 = dot(__x,e1) / dot(e1,e1)
            _a2 = dot(__x,e2) / dot(e2,e2)
            __x -= _a1 * e1 + _a2 * e2

            count, theta = 0, 0.
            for p in range(2,7):
                for i in Parts[p]:
                    _theta = math.atan2(_x[i],_x[i+Nb]) - math.atan2(__x[i],__x[i+Nb])
                    print _theta,
                    if _theta > math.pi:
                        _theta -= 2.*math.pi
                    elif theta < -math.pi:
                        _theta += 2.*math.pi
                    print _theta
                    theta += _theta
                    count += 1
            theta /= count

            Rot = zeros((2,2),float64)
            Rot[0,0] = cos(theta)
            Rot[1,0] = sin(theta)
            Rot[0,1] = -Rot[1,0]
            Rot[1,1] = Rot[0,0]

    ##        for p in range(2,7):
    ##            for i in Parts[p]:
    ##                [__x[i],__x[i+Nb]] = dot(Rot, [__x[i],__x[i+Nb]])
            
            __x += a1 * e1 + a2 * e2
            x = 1. * __x
            _x = 1. * x + dot(dt*A,f)

            print a1, a2, theta


##        D = Jacobian(x, Nb, Links, Tethers)
##        D = array(D.todense())
##
##        e3 = 1. * x
##        e3 = GetEig(D, e1, e2, e3)
##
##        x += e3 * (dot(_x,e3) - dot(x,e3))




        clf()
        scatter(x[:Nb],x[Nb:])
        raw_input("")
        scatter(update_x[:Nb],update_x[Nb:],c='r')
        raw_input("")


        D = Jacobian(x, Nb, Links, Tethers)
        D = array(D.todense())

        F = linalg.solve(dt*A, x-b)
        _f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
        f = F - _f

        e3 = 1. * x
        e3 = GetEig(D, e1, e2, e3)

        f -= e1 * dot(f,e1) / dot(e1,e1)
        f -= e2 * dot(f,e2) / dot(e2,e2)
        f -= e3 * dot(f,e3) / dot(e3,e3)

        y = linalg.solve(D,f)
        y -= e1 * dot(y,e1) / dot(e1,e1)
        y -= e2 * dot(y,e2) / dot(e2,e2)
        y -= e3 * dot(y,e3) / dot(e3,e3)

        x += 1. * y

        f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
        r = b + dot(dt*A,f) - x

        print "Residual!"
        clf(); plot(r); raw_input("")
        MaxRes = max(list(abs(r)))







    _e1 = linalg.solve(dt*A,e1)
    _e2 = linalg.solve(dt*A,e2)
##    _e1, _e2, _e3, F = 0*x, 0*x, 0*x, 0*x    
##    _e1 = MultigridStep(__A,e1,II,_e1,1.)
##    _e2 = MultigridStep(__A,e2,II,_e2,1.)

   
    MaxRes = 1.
    while MaxRes > .01:
        _e1 = MultigridStep(__A,e1,II,_e1,1.)
        _e2 = MultigridStep(__A,e2,II,_e2,1.)

        D = Jacobian(x, Nb, Links, Tethers)
        D = array(D.todense())

        F = linalg.solve(dt*A, x-b)
        #F = MultigridStep(__A,x-b,II,F,1.)
        #F = MultigridStep(__A,x-b,II,F,1.)
        _f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
        #f -= _f
        f = F - _f
        print "F", max(list(abs(F-linalg.solve(dt*A,x-b))))

        # Solve inverse translation eigenvectors
        e3 = 1. * x
        e3 = GetEig(D, e1, e2, e3)

        count = 0
        lam = 0
        E3 = dot(D,e3)
        for q in range(2*Nb):
            if abs(e3[q]) > .0001:
                count += 1
                lam += E3[q] / e3[q]
        lam /= count
        
        print "e3 lam:", lam
        _e3 = linalg.solve(dt*A,e3) - dot(D,e3)
        #_e3 = MultigridStep(__A,e3,II,-_e3,1.)
        print "e3", max(list(abs(_e3-linalg.solve(dt*A,e3))))




##        K = identity(2*Nb,float64)
##        K[0,:] -= e1
##        K[1,:] -= e2
##        K[2,:] -= e3

        K = 1. * D
        K = hstack((K,-matrix(_e1).T,-matrix(_e2).T,-matrix(_e3).T))

        J = identity(2*Nb,float64)
        J = vstack((J,matrix(e1),matrix(e2),matrix(e3)))

        K = dot(J,K)


        __f = 1. * f
        __f -= e1 * dot(__f,e1) / dot(e1,e1)
        __f -= e2 * dot(__f,e2) / dot(e2,e2)
        __f -= e3 * dot(__f,e3) / dot(e3,e3)

        y = linalg.solve(K,hstack((f,dot(f,e1),dot(f,e2),dot(f,e3))))
        Y = y[:2*Nb]
        y[:2*Nb] -= e1 * dot(Y,e1) / dot(e1,e1)
        y[:2*Nb] -= e2 * dot(Y,e2) / dot(e2,e2)
        y[:2*Nb] -= e3 * dot(Y,e3) / dot(e3,e3)        

        x += y[:2*Nb]
        x += y[2*Nb] * e1 + y[2*Nb+1] * e2 + y[2*Nb+2] * e3

        clf(); plot(y); raw_input("")
        clf(); plot(dot(K,y)); raw_input("")
        print dot(K,y)-hstack((f,dot(f,e1),dot(f,e2),dot(f,e3)))
        


######################
######################
######################        
######################        K = zeros((3,3))
######################        K[0,0] = dot(_e1,e1)
######################        K[0,1] = dot(_e2,e1)
######################        K[0,2] = dot(_e3,e1)
######################        K[1,0] = dot(_e1,e2)
######################        K[1,1] = dot(_e2,e2)
######################        K[1,2] = dot(_e3,e2)
######################        K[2,0] = dot(_e1,e3)
######################        K[2,1] = dot(_e2,e3)
######################        K[2,2] = dot(_e3,e3)
######################
######################        lam = 0
######################        if abs(lam) < 1000:
######################            _K = linalg.solve(K,array([dot(f,e1), dot(f,e2), dot(f,e3)]))
######################            x -= e1 * _K[0] + e2 * _K[1] + e3 * _K[2]
######################
########################            f -= e1 * dot(f,e1) / dot(e1,e1)
########################            f -= e2 * dot(f,e2) / dot(e2,e2)
########################            f -= e3 * dot(f,e3) / dot(e3,e3)
######################        else:
######################            _K = linalg.solve(K[:2,:2], array([dot(f,e1), dot(f,e2)]))
######################            x -= e1 * _K[0] + e2 * _K[1]
######################
########################            f -= e1 * dot(f,e1) / dot(e1,e1)
########################            f -= e2 * dot(f,e2) / dot(e2,e2)
######################
######################        # Recalculate eig
######################        D = Jacobian(x, Nb, Links, Tethers)
######################        D = array(D.todense())
######################
######################        e3 = 1. * x
######################        e3 = GetEig(D, e1, e2, e3)
######################
######################        f -= e1 * dot(f,e1) / dot(e1,e1)
######################        f -= e2 * dot(f,e2) / dot(e2,e2)
######################        f -= e3 * dot(f,e3) / dot(e3,e3)
######################            
######################       
######################
######################        y = linalg.solve(D,f)
######################        y -= e1 * dot(y,e1) / dot(e1,e1)
######################        y -= e2 * dot(y,e2) / dot(e2,e2)
######################
######################        print "Rotation:", dot(y,e3) / dot(e3,e3)
######################        #if abs(lam) < .1:
######################        y -= e3 * dot(y,e3) / dot(e3,e3)
######################        print "Rotation:", dot(y,e3) / dot(e3,e3)
######################
######################        clf()
######################        scatter(x[:Nb],x[Nb:])
######################        _x = x + e3
######################        scatter(_x[:Nb],_x[Nb:],c='r')
######################        raw_input("")
######################
######################        
######################        clf(); print "y"; plot(y); plot(e3,c='r'); raw_input("")
######################        clf(); print "Dy"; plot(dot(D,y));
######################        plot(f,c='r'); raw_input("")
######################
######################        x += 1. * y
        

        f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
        r = b + dot(dt*A,f) - x

        print "Residual!"
        clf(); plot(r); raw_input("")
        MaxRes = max(list(abs(r)))



    f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
    r = b + dot(dt*A,f) - x

    print "Residual:", max(list(abs(r)))
        



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
    raw_input("")
