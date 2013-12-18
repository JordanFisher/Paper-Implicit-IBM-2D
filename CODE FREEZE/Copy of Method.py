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
from scipy import sparse, linsolve
import time
import math
#show()


def ModMultigridStep(A, P, L, b, I, x, damp = .3):
    if len(I) <= 2:
        return linalg.solve(A[0],b)
        return x

    r = I[0].T * (b - dot(A[0], x))

    _x = zeros(len(r), float64)
    _x = MultigridStep(A[1:], r, I[1:], _x, damp)

    x += I[0] * _x
    ModVCycle(A, P, L, b, I, x, damp)

    return x
    
def ModVCycle(A, P, L, b, I, x, damp = .3):
    if len(I) <= 1:
        return

    GaussSeidel(len(x), A[0], b, x)

    r = b - dot(A[0], x)
    x[L] += linalg.solve(P,r[L])
    
    r = I[0].T * (b - dot(A[0], x))
    _x = zeros(len(r), float64)
    VCycle(A[1:], r, I[1:], _x)
    x += I[0] * _x

    GaussSeidel(len(x), A[0], b, x)


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

def FastForce(X, Y, Xss, Yss, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData):
    tx = X[LinkIndices[:,1]] - X[LinkIndices[:,0]]
    ty = Y[LinkIndices[:,1]] - Y[LinkIndices[:,0]]

    LinkCurL[:] = (tx**2 + ty**2)**.5

    txl = tx / LinkCurL
    tyl = ty / LinkCurL

    Xss[:], Yss[:] = 0. * Xss, 0. * Yss
    calc = LinkData[:,0] * (tx - LinkData[:,1] * txl)
    IB_c.CollapseSum(Xss, LinkIndices[:,0], calc)
    IB_c.CollapseSum(Xss, LinkIndices[:,1], -calc)    

    calc = LinkData[:,0] * (ty - LinkData[:,1] * tyl)
    IB_c.CollapseSum(Yss, LinkIndices[:,0], calc)
    IB_c.CollapseSum(Yss, LinkIndices[:,1], -calc)    

    Xss[TetherIndex] += TetherData[:,0] * (TetherData[:,1] - X[TetherIndex])
    Yss[TetherIndex] += TetherData[:,0] * (TetherData[:,2] - Y[TetherIndex])    
    
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
    
def Jacobian(z, Nb, Links, Tethers):
    D = zeros((2*Nb,2*Nb),float64)
    for k in range(2*Nb):
        _dF = dF(k, Nb, z[0:Nb], z[Nb:2*Nb], Links, Tethers)
        D[k,:] = _dF
    D = sparse.csc_matrix(D)
    return D

def SeedJacobian(Nb, LinkIndices):
    count = 1
    n = len(LinkIndices[:,0])
    D = zeros((2*Nb,2*Nb),float64)

    for i in range(n): D[LinkIndices[i,0],LinkIndices[i,1]] = count; count += 1
    for i in range(n): D[LinkIndices[i,1],LinkIndices[i,0]] = count; count += 1

    for i in range(n): D[LinkIndices[i,0]+Nb,LinkIndices[i,1]+Nb] = count; count += 1
    for i in range(n): D[LinkIndices[i,1]+Nb,LinkIndices[i,0]+Nb] = count; count += 1
    
    for i in range(n): D[LinkIndices[i,0],LinkIndices[i,1]+Nb] = count; count += 1
    for i in range(n): D[LinkIndices[i,1]+Nb,LinkIndices[i,0]] = count; count += 1

    for i in range(n): D[LinkIndices[i,1],LinkIndices[i,0]+Nb] = count; count += 1
    for i in range(n): D[LinkIndices[i,0]+Nb,LinkIndices[i,1]] = count; count += 1

    for i in range(2*Nb): D[i,i] = count; count += 1
    for i in range(Nb): D[i,i+Nb] = count; count += 1
    for i in range(Nb): D[i+Nb,i] = count; count += 1

    return sparse.csr_matrix(D)

def FastJacobian(z, D, args, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData):
    tx = z[LinkIndices[:,1]] - z[LinkIndices[:,0]]
    ty = z[LinkIndices[:,1] + Nb] - z[LinkIndices[:,0] + Nb]

    l3 = LinkCurL**3
    l1 = 1. / LinkCurL

    dxdx = LinkData[:,0] * (-1. + LinkData[:,1] * (l1 - tx**2 / l3))
    dydy = LinkData[:,0] * (-1. + LinkData[:,1] * (l1 - ty**2 / l3))
    dxdy = -LinkData[:,0] * LinkData[:,1] * tx * ty / l3

    data = 0. * D.data
    n = len(dxdx)
    
    data[:n] -= dxdx
    data[n:2*n] -= dxdx

    data[2*n:3*n] -= dydy
    data[3*n:4*n] -= dydy

    data[4*n:5*n] -= dxdy
    data[5*n:6*n] -= dxdy

    data[6*n:7*n] -= dxdy
    data[7*n:8*n] -= dxdy

    Diag = zeros(Nb,float64)
    IB_c.CollapseSum(Diag, LinkIndices[:,0], dxdx)
    IB_c.CollapseSum(Diag, LinkIndices[:,1], dxdx)
    Diag[TetherIndex] -= TetherData[:,0]
    data[8*n:8*n+Nb] = 1. * Diag

    Diag = zeros(Nb,float64)
    IB_c.CollapseSum(Diag, LinkIndices[:,0], dydy)        
    IB_c.CollapseSum(Diag, LinkIndices[:,1], dydy)
    Diag[TetherIndex] -= TetherData[:,0] 
    data[8*n+Nb:8*n+2*Nb] = 1. * Diag
    
    Diag = zeros(Nb,float64)
    IB_c.CollapseSum(Diag, LinkIndices[:,0], dxdy)        
    IB_c.CollapseSum(Diag, LinkIndices[:,1], dxdy)        
    data[8*n+2*Nb:8*n+3*Nb] = 1. * Diag
    
    Diag = zeros(Nb,float64)
    IB_c.CollapseSum(Diag, LinkIndices[:,1], dxdy)
    IB_c.CollapseSum(Diag, LinkIndices[:,0], dxdy)    
    data[8*n+3*Nb:8*n+4*Nb] = 1. * Diag
    
    D.data[args] = data

    
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
h = Domain_x / 200.
N, M = int(Domain_x / h), int(Domain_y / h)
Domain_x, Domain_y = N * h, M * h
print "N, M =", N, M
dt = .01
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

X, Y, Links, Tethers, Parts = ConstructValve (_s, 0., Domain_x, 0., Domain_y, .02, False)#, True, 1.3,1.7)#.01)
Nb = len(X)

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

# Unpack link and tether data into arrays
LinkIndices = zeros((len(Links),2),int64)
LinkData = zeros((len(Links),2),float64)
LinkCurL = zeros(len(Links),float64)
TetherData = zeros((len(Tethers),3),float64)
TetherIndex = zeros(len(Tethers),int64)
for l in range(len(Links)):
    a, b, s, L = Links[l]
    LinkIndices[l,:] = [a,b]
    LinkData[l,:] = [s, L]
for l in range(len(Tethers)):
    a, x, s = Tethers[l]
    x, y = x
    TetherData[l,:] = [s, x, y]
    TetherIndex[l] = a


#X,Y = 2.*X,2.*Y
##x = zeros(2*Nb)
##x[:Nb],x[Nb:] = cos(X), Y
#x *= 2.
##f = 0. * x
##ff = 0. * f
##
##FastForce(X, Y, Xss, Yss, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)

DSeed = SeedJacobian(Nb, LinkIndices)
SeedArgs = argsort(DSeed.data)

##D1 = zeros((2*Nb,2*Nb),float64)
##QuickJacobian(x, Nb, Links, Tethers, D1)
##
##FastForce(x[:Nb], x[Nb:], Xss, Yss, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
##D = 0. * DSeed
##FastJacobian(x, D, SeedArgs, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
##
##raise

##Time = time.time()
##for i in range(1000):
##    FastForce(X, Y, Xss, Yss, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
##    FastJacobian(x, D1, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
##print time.time() - Time



__I,_x2,_y2,Parts2 = ReduceValve3(X,Y,Parts)
#s2, __I, _x2, _y2, Links2, Tethers2, Parts2 = ReduceValve(_s, X, Y, Links, Tethers, Parts)
Nb2 = len(_x2)


_II = 1. * __I

_n, _m = _II.shape
__II = zeros((2*_n,2*_m),float64)
__II[:_n,:_m] = 1. * _II
__II[_n:,_m:] = 1. * _II
II = [sparse.csc_matrix(__II)]
_x3, _y3, Parts3 = _x2, _y2, Parts2
#s3, _x3, _y3, Links3, Tethers3, Parts3 = s2, _x2, _y2, Links2, Tethers2, Parts2
while len(_II) > 10 and len(II) < 7:
    holdlen = len(_II)
    print holdlen

    _II,_x3,_y3,Parts3 = ReduceValve3(_x3,_y3,Parts3)  
    #s3, _II, _x3, _y3, Links3, Tethers3, Parts3 = ReduceValve(s3, _x3, _y3, Links3, Tethers3, Parts3)
    if len(_II) == holdlen:
        II = II[:len(II)-2]
        break
    _n, _m = _II.shape
    __II = zeros((2*_n,2*_m),float64)
    __II[:_n,:_m] = 1. * _II
    __II[_n:,_m:] = 1. * _II
    II.append(sparse.csc_matrix(__II))



print "Nb =", Nb
hb = 1. / Nb
hb = 1.




PlotValve (X, Y, Links)


i1, i2, i3 = Parts[3][0],Parts[3][0]+Nb,Parts[3][len(Parts[3])-1]
Reduce = zeros((2*Nb-3,2*Nb),float64)
j = 0
for i in range(2*Nb):
    if i != i1 and i != i2 and i != i3:
        Reduce[j,i] = 1.
        j += 1
Reduce = sparse.csr_matrix(Reduce)

l = []
for i in range(Nb):
    if X[i] > .25 and X[i] < .4 and Y[i] < .49 and Y[i] > .31:
            l.extend([i,i+Nb])



A = zeros((2*Nb,2*Nb), float64)
x = zeros(2*Nb,float64)
f = zeros(2*Nb,float64)
_f = zeros(2*Nb,float64)
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

    P = dt * A[l,:][:,l]

  
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
    xp = x + dt*dot(A,f)

    for i in range(4):
        x = x + e1 * dot(xp-x,e1)/dot(e1,e1) + e2 * dot(xp-x,e2)/dot(e2,e2)

        center = e1*dot(x,e1)/dot(e1,e1) + e2*dot(x,e2)/dot(e2,e2) 

        z = x - center
        e3 = 0. * x
        for p in range(9,10):#range(2,7):
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

    # Solve for rotational eigenvalue
    e3 = 0. * x
    z = x - e1*dot(x,e1)/dot(e1,e1) - e2*dot(x,e2)/dot(e2,e2)
    for p in range(2,7):
        for i in Parts[p]:
            e3[i] = z[i+Nb]
            e3[i+Nb] = -z[i]

    


    _e1 = ModMultigridStep(__A,P,l,e1,II,_e1,1.)
    _e2 = ModMultigridStep(__A,P,l,e2,II,_e2,1.)
    _e3 = ModMultigridStep(__A,P,l,e3,II,_e3,1.)    

    ChangeX = 1.   
    MaxRes = 1.
    count = 0
    F = 0. * F
    while count < 7:
        holdx = 1. * x
        count += 1

##        _e1 = MultigridStep(__A,e1,II,_e1,1.)
##        _e2 = MultigridStep(__A,e2,II,_e2,1.)
##        _e1 = linalg.solve(dt*A,e1)
##        _e2 = linalg.solve(dt*A,e2)       

        F = ModMultigridStep(__A,P,l,x-b,II,F,1.)
#        F = MultigridStep(__A,x-b,II,F,1.)
#        F = linalg.solve(dt*A, x-b)
        _f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
        f = F - _f

        # Solve for rotational eigenvalue
        e3 = 0. * x
        z = x - e1*dot(x,e1)/dot(e1,e1) - e2*dot(x,e2)/dot(e2,e2)
        for p in range(2,7):
            for i in Parts[p]:
                e3[i] = z[i+Nb]
                e3[i+Nb] = -z[i]


#        _e3 = linalg.solve(dt*A,e3) #- Dsp*e3
#        _e3 = MultigridStep(__A,e3,II,_e3,1.)
#        _e3 = ModMultigridStep(__A,P,l,e3,II,_e3,1.)
        
        K = zeros((3,3))
        K[0,0] = dot(_e1,e1)
        K[0,1] = dot(_e2,e1)
        K[0,2] = dot(_e3,e1)
        K[1,0] = dot(_e1,e2)
        K[1,1] = dot(_e2,e2)
        K[1,2] = dot(_e3,e2)
        K[2,0] = dot(_e1,e3)
        K[2,1] = dot(_e2,e3)
        K[2,2] = dot(_e3,e3)

        #_K = linalg.solve(K,array([dot(f,e1), dot(f,e2), dot(f,e3)]))
        _K = linalg.solve(K,array([dot(F,e1), dot(F,e2), dot(F,e3)]))
        if True:
            x -= e1 * _K[0] + e2 * _K[1]# + e3 * _K[2]

            center = e1*dot(x,e1)/dot(e1,e1) + e2*dot(x,e2)/dot(e2,e2) 
            z = x - center
            e3 = 0. * x

            theta = -_K[2]
            Rot = zeros((2,2),float64)
            Rot[0,0] = cos(theta)
            Rot[1,0] = -sin(theta)
            Rot[0,1] = -Rot[1,0]
            Rot[1,1] = Rot[0,0]

            for p in range(2,7):
                for i in Parts[p]:
                    [z[i],z[i+Nb]] = dot(Rot, [z[i],z[i+Nb]])
            x = z + center

            print "  ", _K
        else:
            print "Skip this first step"






        subMax = 1000.
        damp = 1.
#        clf(); scatter(x[:Nb],x[Nb:]); raw_input("")
        while subMax > .00001:
#            clf(); scatter(x[:Nb],x[Nb:]); raw_input("")

            Time = time.time()
            
            FastForce(x[:Nb], x[Nb:], _f[:Nb], _f[Nb:], LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
            D = 0. * DSeed
            FastJacobian(x, D, SeedArgs, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
            f = F - _f
##
##
##            
######            D1 = 0. * D.todense()
######            QuickJacobian(x, Nb, Links, Tethers, D1)
######
######            raise
######            D1 = sparse.csr_matrix(D)
##
##            print "t1:", Time-time.time(),
##            Time = time.time()
##
##            Dsp = sparse.csr_matrix(array(D))
##
##            print "t2:", Time-time.time(),
##            Time = time.time()
##
##            _f = _FiberForce (Nb, x, Xss, Yss, Links, Tethers)
##            f = F - _f
##
##            print "t3:", Time-time.time(),
##            Time = time.time()

            e3 = 0. * x
            z = x - e1*dot(x,e1)/dot(e1,e1) - e2*dot(x,e2)/dot(e2,e2)
            for p in range(2,7):
                for i in Parts[p]:
                    e3[i] = z[i+Nb]
                    e3[i+Nb] = -z[i]

##            print "t4:", Time-time.time()
##            Time = time.time()


            f -= e1 * dot(f,e1) / dot(e1,e1)
            f -= e2 * dot(f,e2) / dot(e2,e2)
            f -= e3 * dot(f,e3) / dot(e3,e3)
            subMax = max(list(abs(f)))
           
            #y = linsolve.spsolve(Dsp,f)
            #y = linalg.solve(array(D),f)

            ReduceD = Reduce * D * Reduce.T
#            ReduceD = Reduce * Dsp * Reduce.T
#            Dlu = scipy.linsolve.splu(Dsp)
            Dlu = scipy.linsolve.splu(ReduceD)
            #y = Dlu.solve(f)
            y = Dlu.solve(Reduce*f)
            y = Reduce.T * y

            y -= e1 * dot(y,e1) / dot(e1,e1)
            y -= e2 * dot(y,e2) / dot(e2,e2)
            y -= e3 * dot(y,e3) / dot(e3,e3)
#            if subMax > .1: y -= e3 * dot(y,e3) / dot(e3,e3)
            
            x += damp * y
            
            print time.time() - Time

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

#    f = linalg.solve(dt*A,x-b)
#    Xss, Yss = 1.*f[:Nb], 1.*f[Nb:]

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
