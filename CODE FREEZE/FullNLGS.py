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
from scipy import sparse, linsolve
import time
import math
#show()

def GetEigs(B):
    eigs, vecs = Eig(B)
    args = argsort(array(abs(eigs)))
    return eigs[args], vecs[args]

def EllipseCircumference (r1, r2):
    return math.pi * ( 3.*(r1 + r2) - ((3.*r1+r2)*(r1+3.*r2))**.5)


def PCG(A,b,x0):
    print "__------>", A[0,0]
    r = b - dot(A,x0)
    z = 1. * r / A.diagonal() ##
    d = 1. * z

    x = x0

    for i in range(len(b)):
        a = dot(z,r) / dot(d,dot(A,d))
        x = x + a * d
        ZR = dot(z,r)
        r = r - a*dot(A,d)
        z = 1. * r / A.diagonal() ##
        B = dot(z,r) / ZR
        d = z + B*d

        print "CG step ", i, dot(r,r)**.5        
        if i > 9 and dot(r,r)**.5 < .2:
            return x
    

def CG(A,b,x0):
    r = b - dot(A,x0)

    print "CG pre ", dot(r,r)**.5        
    if dot(r,r)**.5 < 1e-4:
        return x0

    w = -r
    z = dot(A,w)
    a = dot(r,w) / dot(w,z)
    x = x0 + dot(a,w)
    B = 0
  
    for i in range(len(b)):
        r = r - dot(a,z)
        print "CG step ", i, dot(r,r)**.5        
        if i > 9 and dot(r,r)**.5 < 5e-3:
            return x

        B = dot(r,z) / dot(w,z)
        w = -r + dot(B,w)
        z = dot(A,w)
        a = dot(r,w) / dot(w,z)
        x = x + dot(a,w)

    return x


def EzSparse(N,M,A):
    IShell = zeros((N,3),float64)
    IShelli = zeros((N,3),int32)
    IShelld = zeros(N,int32)

    IB_c.EzSparse(N,M,A,IShell,IShelli,IShelld)

    return (N, M, max(list(IShelld)), IShell, IShelli, IShelld)

def EzSparseMM(A1, A2, newB = True, B = -1):
    N, M, s, IShell, IShelli, IShelld = A1

    _N, _M = A2.shape

    if M != _N:
        print N, M
        print A2.shape
        raise("Incompatible dimensions")

    if newB:
        B = zeros((N,_M),float64)

    IB_c.SparseMM(N, M, _M, s, IShell, IShelli, IShelld, A2, B)

    return B


def Jacobi(A, b, x):
    x += (b - dot(A,x)) / A.diagonal()
    

def FastVCycle(A, b, I, x, damp = .3):
    if len(I) <= 1:
        return

    #Jacobi(A[0], b, x)
    #GaussSeidel(len(x), A[0], b, x)
    IB_c.GaussSeidel(len(x), A[0], b, x, damp)
    
    r = I[0].T * (b - dot(A[0], x))
    _x = zeros(len(r), float64)
    FastVCycle(A[1:], r, I[1:], _x) 
    x += I[0] * _x


def FastMultigridStep(A, b, I, x, damp = .3):
    if len(I) <= 2:
        return linalg.solve(A[0],b)
        return x

    r = I[0].T * (b - dot(A[0], x))

    _x = zeros(len(r), float64)
    _x = FastMultigridStep(A[1:], r, I[1:], _x, damp)

    x += I[0] * _x
    FastVCycle(A, b, I, x, damp)

    return x


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
    Nb = len(X)
    
    tx = X[LinkIndices[:,1]] - X[LinkIndices[:,0]]
    ty = Y[LinkIndices[:,1]] - Y[LinkIndices[:,0]]

    LinkCurL[:] = (tx**2 + ty**2)**.5

    txl = tx / LinkCurL
    tyl = ty / LinkCurL

    for i in range(Nb):
        if LinkCurL[i] == 0:
            txl[i] = tyl[i] = 0.

    Xss[:], Yss[:] = 0. * Xss, 0. * Yss
    calc = LinkData[:,0] * (tx - LinkData[:,1] * txl)
    IB_c.CollapseSum(Xss, LinkIndices[:,0], calc)
    IB_c.CollapseSum(Xss, LinkIndices[:,1], -calc)    

    calc = LinkData[:,0] * (ty - LinkData[:,1] * tyl)
    IB_c.CollapseSum(Yss, LinkIndices[:,0], calc)
    IB_c.CollapseSum(Yss, LinkIndices[:,1], -calc)    

    Xss[TetherIndex] += TetherData[:,0] * (TetherData[:,1] - X[TetherIndex])
    Yss[TetherIndex] += TetherData[:,0] * (TetherData[:,2] - Y[TetherIndex])    

def PointForce (i, Nb, X, Y, Links, Tethers):
    fx, fy = 0., 0.

    for A in Links:
        a, b, s, L = A

        if a == i or b == i:
            l = ((X[b] - X[a])**2 + (Y[b] - Y[a])**2)**.5
            tx, ty = (X[b] - X[a]) / l, (Y[b] - Y[a]) / l

            if a == i:
                fx += s * (X[b] - X[a] - L * tx)
                fy += s * (Y[b] - Y[a] - L * ty)
            else:
                fx -= s * (X[b] - X[a] - L * tx)
                fy -= s * (Y[b] - Y[a] - L * ty)
                
    
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

def FastJacobian2(z, D, args, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData):
    tx = z[LinkIndices[:,1]] - z[LinkIndices[:,0]]
    ty = z[LinkIndices[:,1] + Nb] - z[LinkIndices[:,0] + Nb]

    l3 = LinkCurL**3
    l1 = 1. / LinkCurL

    dxdx = (-1. + LinkData[:,1] * (l1 - tx**2 / l3))
    dydy = (-1. + LinkData[:,1] * (l1 - ty**2 / l3))
    dxdy = -LinkData[:,1] * tx * ty / l3

    for i in range(Nb):
        if LinkCurL[i] == 0:
            dxdx[i] = -1.
            dydy[i] = -1.
            dxdy[i] = 0.
        print i, LinkCurL[i], dxdx[i], dxdy[i], dydy[i]            
    
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
    Diag[TetherIndex] -= 1.#TetherData[:,0]
    data[8*n:8*n+Nb] = 1. * Diag

    Diag = zeros(Nb,float64)
    IB_c.CollapseSum(Diag, LinkIndices[:,0], dydy)        
    IB_c.CollapseSum(Diag, LinkIndices[:,1], dydy)
    Diag[TetherIndex] -= 1.#TetherData[:,0] 
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
  


N = 256
Domain_x, Domain_y = 1., 1.
h = Domain_x / N

CurT = 0
T = .5

Nb = 2 * N
hb = 1. / Nb
_s = 1000000. / hb**2
dt = .00001


WideLambda = zeros((N,N),float64)
ShortLambda = zeros((N,N),float64)
IB_c.InitWideLaplacian(N, N, h, WideLambda)
IB_c.InitShortLaplacian(N, N, h, ShortLambda)
DxSymbol = InitDxSymbol(N, N, h)
DySymbol = InitDySymbol(N, N, h)

_h = h
_N, _M = N, N
UField, VField, UField2, VField2 = InitVelField(_N, _M, _h, h, dt)

print UField[0,0],VField2[0,0]
clf()
clf(); imshow(UField); colorbar(); #show(); #&&raw_input("")
clf(); imshow(VField2); colorbar(); #show(); #&&raw_input("")

#Nb = 6
X = zeros(Nb, float64)
Y = zeros(Nb, float64)

Links, Tethers = [], []
Ellipse (Nb, X, Y, .4, .2, .5, .5)
l = .9 * EllipseCircumference (.4, .2) / Nb

LinksFake = []
for i in range(Nb-1):
#    l = 1.1*((X[i]-X[i+1])**2+(Y[i]-Y[i+1])**2)**.5
#    print ((X[i]-X[i+1])**2+(Y[i]-Y[i+1])**2)**.5, l
    Links.append( (i, i+1, _s, l) )
    LinksFake.append( (i, i+1, _s, 0.) )
#l = 1.1*((X[Nb-1]-X[0])**2+(Y[Nb-1]-Y[0])**2)**.5
Links.append( (Nb-1, 0, _s, l) )
LinksFake.append( (Nb-1, 0, _s, 0.) )
##Tethers.append( (0, (X[0],Y[0]), _s) ) 
##Tethers.append( (1, (X[1],Y[1]), _s) )
##Tethers.append( (2, (X[2],Y[2]), _s) ) 
##for i in range(Nb):
##    Tethers.append( (i, (X[i],Y[i]), .03*_s) )
#    Tethers.append( (i, (.5,.5), .1*_s) )


############ Coarse ellipse ##############
Nb2 = Nb / 2
hb2 = 1. / Nb2
_s2 = 1000000. / hb2**2

X2 = zeros(Nb2, float64)
Y2 = zeros(Nb2, float64)

Links2, Tethers2 = [], []
Ellipse (Nb2, X2, Y2, .4, .2, .5, .5)
l = .3 * EllipseCircumference (.4, .2) / Nb2

for i in range(Nb2-1):
#    l = ((X[i]-X[i+1])**2+(Y[i]-Y[i+1])**2)**.5
#    print ((X[i]-X[i+1])**2+(Y[i]-Y[i+1])**2)**.5, l
    Links2.append( (i, i+1, _s2, l) )
#l = ((X[Nb-1]-X[0])**2+(Y[Nb-1]-Y[0])**2)**.5
Links2.append( (Nb2-1, 0, _s2, l) )
##Tethers.append( (0, (X[0],Y[0]), _s) ) 
##Tethers.append( (1, (X[1],Y[1]), _s) )
##Tethers.append( (2, (X[2],Y[2]), _s) ) 
##for i in range(Nb2):
##    Tethers2.append( (i, (X2[i],Y2[i]), _s2) )
##    Tethers.append( (i, (.5,.5), .1*_s) )




Xs = zeros(Nb, float64)
Ys = zeros(Nb, float64)
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

gridx = zeros((N,N),float64)
gridy = zeros((N,N),float64)
for j in range(N):
    gridx[j,:] = Domain_x * float(j) / N
for j in range(N):
    gridy[:,j] = Domain_y * float(j) / N

# Unpack link and tether data into arrays
LinkIndices = zeros((len(Links),2),int64)
LinkData = zeros((len(Links),2),float64)
LinkDataFake = zeros((len(Links),2),float64)
LinkCurL = zeros(len(Links),float64)
TetherData = zeros((len(Tethers),3),float64)
TetherIndex = zeros(len(Tethers),int64)
for l in range(len(Links)):
    a, b, s, L = Links[l]
    LinkIndices[l,:] = [a,b]
    LinkData[l,:] = [s, L]
    LinkDataFake[l,:] = [s, L / 2.]
for l in range(len(Tethers)):
    a, x, s = Tethers[l]
    x, y = x
    TetherData[l,:] = [s, x, y]
    TetherIndex[l] = a


######### Coarse unpack ###############
# Unpack link and tether data into arrays
LinkIndices2 = zeros((len(Links2),2),int64)
LinkData2 = zeros((len(Links2),2),float64)
LinkCurL2 = zeros(len(Links2),float64)
TetherData2 = zeros((len(Tethers2),3),float64)
TetherIndex2 = zeros(len(Tethers2),int64)
for l in range(len(Links2)):
    a, b, s, L = Links2[l]
    LinkIndices2[l,:] = [a,b]
    LinkData2[l,:] = [s, L]
for l in range(len(Tethers2)):
    a, x, s = Tethers2[l]
    x, y = x
    TetherData2[l,:] = [s, x, y]
    TetherIndex2[l] = a




DSeed = SeedJacobian(Nb, LinkIndices)
SeedArgs = argsort(DSeed.data)


I = []
Isp = []
IspT = []
__N = Nb
while __N > 2 and __N%2==0:
    _I = zeros((2*__N,__N), float64)
    for i in range(__N/2):
        _I[2*i,i] = 1.
        _I[2*i+1,(i+1)%(__N/2)] = .5
        _I[2*i+1,(i)%(__N/2)] = .5

        _I[2*i+__N,i+__N/2] = 1.
        _I[2*i+1+__N,(i+1)%(__N/2)+__N/2] = .5
        _I[2*i+1+__N,(i)%(__N/2)+__N/2] = .5

    I.append(sparse.csc_matrix(_I))
    Isp.append(EzSparse(2*__N,__N,_I))
    IspT.append(EzSparse(__N,2*__N,_I.T))
    __N /= 2


##i1, i2, i3 = 1000,1000,1000
##Reduce = zeros((2*Nb,2*Nb),float64)
i1, i2, i3 = 0, Nb, Nb+1000
Reduce = zeros((2*Nb-2,2*Nb),float64)
j = 0
for i in range(2*Nb):
    if i != i1 and i != i2 and i != i3:
        Reduce[j,i] = 1.
        j += 1
Reduce = sparse.csr_matrix(Reduce)



A = zeros((2*Nb,2*Nb), float64)
x = zeros(2*Nb,float64)
f = zeros(2*Nb,float64)
_f = zeros(2*Nb,float64)
_e1, _e2, _e3, F = 0*x, 0*x, 0*x, 0*x
e1, e2, e3 = 0. * x, 0. * x, 0. * x

for i in range(Nb):
    e1[i] = 1.
    e2[i+Nb] = 1.

Old_X, Old_Y = 1. * X, 1. * Y

b = zeros(2*Nb,float64)
count = 0
debug = False
ShowStats = True
ExplicitPredict = False
TotalTime = 0
TotalCount = 0
while CurT < T:
    Time = time.time()

    Predict_X, Predict_Y = X + .5 * (X - Old_X), Y + .25 * (Y - Old_Y)

    Old_u, Old_v = 1. * u, 1. * v
    Old_X, Old_Y = 1. * X, 1. * Y

    IB_c.CentralDerivative_x (N, N, h, u, ux)
    IB_c.CentralDerivative_x (N, N, h, v, vx)
    IB_c.CentralDerivative_y (N, N, h, u, uy)
    IB_c.CentralDerivative_y (N, N, h, v, vy)

##    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, 0., 0.)
    fx, fy = ExplicitTerms (dt, u, v, 0., 0., 0., 0., 0., 0.)

    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, 1., X, Y, u, v)

    if ShowStats:
        print "Explicit:", time.time() - Time

    A = zeros((2*Nb,2*Nb),float64)
    IB_c.ComputeMatrix (X, Y, _N, _M, Nb, _h, hb, UField, VField, UField2, VField2, A, -1)#N/6)

    if ShowStats:
        print "Compute A:", time.time() - Time

    __A = [dt*A]
    for i in range(len(I)):
        __A.append( EzSparseMM(IspT[i],EzSparseMM(IspT[i], __A[i].T).T) )

    if ShowStats:
        print "Init Multi:", time.time() - Time


    b[:Nb], b[Nb:] = X + dt*Xvel, Y + dt*Yvel
    x[0:Nb], x[Nb:2*Nb] = 1. * X, 1. * Y


    
    

##########################################################################################
##########################################################################################
##########################################################################################
######################################## Newton Solve ####################################
##########################################################################################
##########################################################################################
##########################################################################################

    
    NewtonRes = 100
    count = 0
    while NewtonRes > .000000000001:# and count < 50:
        count += 1

####        if count % 1 == 0:
####            print "new"
####            Tethers = []
####            for i in range(Nb):
####                Tethers.append( (i, (x[i],x[i+Nb]), .01*_s) )
####
####            TetherData = zeros((len(Tethers),3),float64)
####            TetherIndex = zeros(len(Tethers),int64)
####            for l in range(len(Tethers)):
####                a, x_loc, s = Tethers[l]
####                x_coor, y_coor = x_loc
####                TetherData[l,:] = array([s, x_coor, y_coor])
####                TetherIndex[l] = a




        
        FastForce(x[:Nb], x[Nb:], _f[:Nb], _f[Nb:], LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
        r = (x-dot(__A[0],_f)) - b
        D = Jacobian(x, Nb, Links, Tethers).todense()
        DFake = Jacobian(x, Nb, LinksFake, Tethers).todense()
        D = 1. * D + .0 * DFake

##        eigsS, vecsS = GetEigs(dot(__A[0], 10.*identity(2*Nb)/__A[0][0,0]-D))
##        print eigsS
##        raw_input("")



######        eigs, vecs = Eig(dot(__A[0],D))
######        eigs = eigs
######        eigs.sort()
########        print eigs
########        print "eigs"
######        print "Eigs!:", eigs[0],eigs[2*Nb-1]
######
######        clf()
######        plot(1.*eigs)
######
######
######        eigs, vecs = Eig(dot(D,__A[0]))
######        eigs = eigs
######        eigs.sort()
########        print eigs
########        print "eigs"
######        print "Eigs!:", eigs[0],eigs[2*Nb-1]
######
######        clf()
######        plot(eigs,c='r')
######        raw_input("")


########        print D
########        print D.diagonal()
########        print "Eigs!"
########        clf()
########        quiver(x[:Nb],x[Nb:],vecs[:Nb,0], vecs[Nb:,0])
########        raw_input("")
########        clf()
########        quiver(x[:Nb],x[Nb:],vecs[:Nb,1], vecs[Nb:,1])        
####        raw_input("")



        P = identity(2*Nb) - __A[0] * D
        x -= 1. * linalg.solve(P,r)

####        P = array(P)
####        M = array(dot(__A[0],D))
####        l = .5
####
####        z = 0. * x
####        #inva = inv(__A[0])[0,0]
####        #print "inva", inva, 1. / __A[0][0,0]
####        alpha = __A[0][0,0]
####        for i in range(1):
######            z = l * (r + dot(M, z)) + (1. - l) * z
####
####            R = linalg.solve(__A[0], r - z) + 1. * z / alpha
####            _P = 1. * identity(2*Nb) / alpha - D
####            z = l * linalg.solve(_P,R) + (1. - l) * z
####            
####
####            print i, max(abs(dot(P,z) - r))
####
####        x -= z            


##        x += .001*dot(__A[0],_f)



####        zz = linalg.solve(P,r)
####        __P = [array(P)]
####        for i in range(len(I)):
####            __P.append( EzSparseMM(IspT[i],EzSparseMM(IspT[i], __P[i].T).T) )
####
####        z = zeros(2*Nb)
#####        z = FastMultigridStep(__P,r,I,z,.02)
#####        x -= z
####
####        clf()
####        for i in range(5):
####            z = FastMultigridStep(__P,r,I,z,.3)
######            x -= .2 * r / array(P).diagonal()
######            plot(x)
####            plot(z,c='b')
####            plot(zz,c='r')
####            raw_input("")
####
####        x -= z





##_dF = dF(k, Nb, z[0:Nb], z[Nb:2*Nb], Links, Tethers)
##PointForce (i, Nb, X, Y, Links, Tethers)


########        holdx = 1. * x
########        newx = 1. * x
########        for i in range(Nb):
########            print i
########            lilres = 100
########            #holdx = x[i]
########            x = 1. * holdx
########            while lilres > .0001:
############                FastForce(x[:Nb], x[Nb:], _f[:Nb], _f[Nb:], LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
############                r = (x[i] - dot(__A[0][i,:], _f)) - b[i]
############                _J = dF(i, Nb, x[:Nb], x[Nb:], Links, Tethers)
############                x[i] -= .75 * r / (1. - dot(__A[0][i,:], _J))
############                lilres = abs(r)
########
########
########                FastForce(x[:Nb], x[Nb:], _f[:Nb], _f[Nb:], LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
########                rx = (x[i] - dot(__A[0][i,:], _f)) - b[i]
########                ry = (x[i+Nb] - dot(__A[0][i+Nb,:], _f)) - b[i+Nb]
########                _Jx = dF(i, Nb, x[:Nb], x[Nb:], Links, Tethers)
########                _Jy = dF(i+Nb, Nb, x[:Nb], x[Nb:], Links, Tethers)
########                J = zeros((2,2),float64)
########                J[0,0] = 1. - dot(__A[0][i,:], _Jx)
########                J[1,0] = -dot(__A[0][i+Nb,:], _Jx)
########                J[1,1] = 1. - dot(__A[0][i+Nb,:], _Jy)
########                J[0,1] = -dot(__A[0][i,:], _Jy)
########
########                z = .75*linalg.solve(J, array([rx, ry]))
########                x[i] -= z[0]
########                x[i+Nb] -= z[1]
########
########                lilres = max(abs(array([rx,ry])))
########                print "  ", lilres, J[0,0], J[0,1], J[1,0], J[1,1]
########
########            newx[i], newx[i+Nb] = x[i], x[i+Nb]
########            
########                #print _J
########                #raw_input("")
########            #x[i] = .5 * x[i] + .5 * holdx
########        x = 1. * newx + .0 * holdx


##        zz = linalg.solve(P,r)
##        x -= linalg.solve(P,r)

        #clf()
        #scatter(x[:Nb],x[Nb:])


####        Tethers = []
####        TetherData = zeros((len(Tethers),3),float64)
####        TetherIndex = zeros(len(Tethers),int64)


        FastForce(x[:Nb], x[Nb:], _f[:Nb], _f[Nb:], LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
        r = (x-dot(__A[0],_f)) - b

        NewtonRes = max(abs(r))
        print "Residual:", NewtonRes, x[0]

        if abs(x[0]) > 10 and NewtonRes < 1:
            raise
#        print
#        raw_input("")



    FastForce(x[:Nb], x[Nb:], f[:Nb], f[Nb:], LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
    r = b + dot(dt*A,f) - x

    if ShowStats:
        print "Residual:", max(list(abs(r)))
##    print
##    print
        

    X, Y = 1.*x[0:Nb], 1.*x[Nb:2*Nb]

    u, v = 1. * Old_u, 1. * Old_v
##    IB_c.CentralDerivative_x (N, M, h, u, ux)
##    IB_c.CentralDerivative_x (N, M, h, v, vx)
##    IB_c.CentralDerivative_y (N, M, h, u, uy)
##    IB_c.CentralDerivative_y (N, M, h, v, vy)

    Xss, Yss = FiberForce (Nb, X, Y, Xss, Yss, Links, Tethers)   

    fx, fy = ForceToGrid (N, N, h, Nb, hb, Old_X, Old_Y, Xss, Yss)
    
##    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)
    fx, fy = ExplicitTerms (dt, u, v, 0., 0., 0., 0., fx, fy)
    
    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)

    u = ifft2(u).real
    v = ifft2(v).real

    Xvel, Yvel = VelToFiber (N, N, h, Nb, hb, 1., Old_X, Old_Y, u, v)

    Time = time.time() - Time
    TotalTime += Time
    TotalCount += 1

    Ts = [.01,.02,.05,.1,.5]
    for _T in Ts:
        if CurT < _T and CurT + dt >= _T:
            print "Time", _T, ":", count, TotalTime, TotalTime / TotalCount
    #print "Time:", CurT
    CurT += dt


    print CurT, "Implicit:", Time
    
    count += 1
    if TotalCount % 1 == 0:
        P = ifft2(P).real
        clf()
        
##        quiver(gridx[::5,::5],gridy[::5,::5],u[::5,::5],v[::5,::5],
##               pivot = 'middle', units = 'height', width = .003, scale = 30)
        #imshow(u)        
        #imshow((vx - uy).T, origin='lower',extent=[0,2,0,1])
        #colorbar()

        #Contour = contourf((vx - uy).T, [70,30,-30,-70],origin='lower',extent=[0,2,0,1])
        Contour = contourf((vx - uy).T,[100,70,30,-30,-70,-100],origin='lower',extent=[0,1,0,1])
        colorbar()
        #clabel(Contour)
        scatter(X,Y,c='k',s=3)
        #quiver(X,Y,f[:Nb],f[Nb:])
        #scatter(M*Y,N*X/2,c='k')
        #quiver(M*Y,N*X/2,Y-Old_Y,-(X-Old_X))
        axis([0, 1, 0, 1])
        #savefig("plots\\fig_c"+str(TotalCount)+"_N"+str(N)+".pdf",format='pdf')
        #show()
        #raw_input("")
    print count






