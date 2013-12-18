import sys
sys.path.append("c:\HeartValve")
sys.path.append("c:\HeartValve\IB_c\Release")
from pylab import *
from numpy import *
from numpy.fft import fft2, ifft2, fft, ifft, rfft
from IB_Methods import *
import IB_c
from ValveSetup import *
from scipy import sparse
import time
plot([1])
show()

def ComputeTensionMatrix (Nb, X, Y, Links, Tethers):
    M = zeros((2*Nb,2*Nb),float64)
    for A in Links:
        a, b, s, L = A
        l = ((X[b] - X[a])**2 + (Y[b] - Y[a])**2)**.5
        _s = s * (l - L) / l

        M[a,b] += _s
        M[b,a] += _s
        M[a,a] -= _s
        M[b,b] -= _s
        
    for A in Links:
        a, b, s, L = A
        M[a+Nb,b+Nb] = M[a,b]
        M[b+Nb,a+Nb] = M[b,a]
        M[a+Nb,a+Nb] = M[a,a]
        M[b+Nb,b+Nb] = M[b,b]

    for A in Tethers:
        a, _x, s = A
        x, y = _x

        M[a,a] -= s
        M[a+Nb,a+Nb] -= s
        
    return sparse.csc_matrix(M)


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
    
 


  


Domain_x, Domain_y = 2., 1.
h = .025#.007
N, M = int(Domain_x / h), int(Domain_y / h)
print N,M
print
dt = .000005
T = .005
CurT = 0

_s = 1000000000.

WideLambda = zeros((N,M),float64)
ShortLambda = zeros((N,M),float64)
IB_c.InitWideLaplacian(N, M, h, WideLambda)
IB_c.InitShortLaplacian(N, M, h, ShortLambda)
DxSymbol = InitDxSymbol(N, M, h)
DySymbol = InitDySymbol(N, M, h)

UField, VField = InitVelField(N, M, h, h, dt)

X, Y, Links, Tethers = ConstructValve (_s, 0., 2., 0., 1., .03)#.01)
Nb = len(X)
print Nb
print
hb = 1. / Nb
##for j in range(Nb):
##    X[j] += .002*cos(j)
##    Y[j] += .002*sin(j**2)

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


PlotValve (X, Y, Links)



I = InitMultigridIMatrices(Nb)
A = zeros((2*Nb,2*Nb), float64)


b = zeros(2*Nb,float64)
count = 0
while CurT < T:
    Old_X, Old_Y = 1. * X, 1. * Y
    print "Time:", CurT
    CurT += dt

    # GET TIME
    t1 = time.clock()

    Old_u, Old_v = 1. * u, 1. * v
    Old_X, Old_Y = 1. * X, 1. * Y

    IB_c.CentralDerivative_x (N, M, h, u, ux)
    IB_c.CentralDerivative_x (N, M, h, v, vx)
    IB_c.CentralDerivative_y (N, M, h, u, uy)
    IB_c.CentralDerivative_y (N, M, h, v, vy)

    Xss, Yss = FiberForce (Nb, X, Y, Xss, Yss, Links, Tethers)
   
    fx, fy = ForceToGrid (N, M, h, Nb, hb, Old_X, Old_Y, Xss, Yss)
    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)
    fx += 1.*ones((N,M),float64)

    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)
    u = ifft2(u).real
    v = ifft2(v).real

    t2 = time.clock()
    Xvel, Yvel = VelToFiber (N, M, h, Nb, hb, 1., X, Y, u, v)
    print "veltofiber", time.clock() - t2

    # PRINT TIME
    print "First fluid solve", time.clock() - t1


    # GET TIME
    t1 = time.clock()
    A = zeros((2*Nb,2*Nb),float64)
    IB_c.ComputeMatrix (X, Y, N, M, Nb, h, hb, UField, VField, A, int(N/50))#, band = Nb/8+1)
    Asp = sparse.csc_matrix(A)
    print "Matrix computed", time.clock() - t1

    # GET TIME
    t1 = time.clock()
    IB_c.ComputeMatrix (X, Y, N, M, Nb, h, hb, UField, VField, A, -1)#, band = Nb/8+1)
    Asp2 = sparse.csc_matrix(A)
    print "Matrix computed long", time.clock() - t1
    

    # GET TIME
    t1 = time.clock()
    
    for i in range(Nb):
        b[i] = dt * Xvel[i]
        b[i+Nb] = dt * Yvel[i]

    gaussx = zeros(2*Nb,float64)
#    gaussx[0:Nb], gaussx[Nb:2*Nb] = 1. * X, 1. * Y
    for q in range(1):
        print q
        t2 = time.clock()
        DssMatrix = ComputeTensionMatrix (Nb, X+gaussx[0:Nb], Y+gaussx[Nb:2*Nb], Links, Tethers)
        print "Dss", time.clock() - t2

        t2 = time.clock()
        _A = identity(2*Nb,dtype=float64) - array(dt * (Asp * DssMatrix).todense())
        print "_A", time.clock() - t2

#        _gaussx = 1. * linalg.solve (_A, b)# + .8 * gaussx

        t2 = time.clock()
        # Multigrid system matrices
        __A = [_A]
        _N = Nb/2
        i = 0
        for i in range(len(I)):
            __A.append( array((I[i].T * matrix(__A[i]) * I[i]).todense()) )
        print "__TIME__:", time.clock() - t2

        for q in range(2):
            t2 = time.clock()
            MultigridStep(__A, b, I, gaussx)
            print "__time__:", time.clock() - t2
##            clf()
##            scatter(gaussx[0:Nb],gaussx[Nb:2*Nb])
##            scatter(_gaussx[0:Nb],_gaussx[Nb:2*Nb],c='r')
##            draw()
##            raw_input("")


        
#        print gaussx[0:5]
#        X += gaussx[0:Nb]
#        Y += gaussx[Nb:2*Nb]
#        x = zeros(2*Nb,float64)
#        x[0:Nb], x[Nb:2*Nb] = X, Y
#        b += dt*dot(A,dot(DssMatrix,gaussx))
#        DssMatrix = ComputeTensionMatrix (Nb, gaussx[0:Nb], gaussx[Nb:2*Nb], Links, Tethers)

##        clf()
##        scatter(X+gaussx[0:Nb],Y+gaussx[Nb:2*Nb])
##        draw()
##        raw_input("")
    X, Y = X+gaussx[0:Nb], Y+gaussx[Nb:2*Nb]
    print "System solved", time.clock() - t1


##########    A = identity(2*Nb,dtype=float64) - dt * dot(A, DssMatrix)
##########    gaussx = linalg.solve (A, b)
##########
##########    # Multigrid system matrices
##########    _A = [A]
##########    _N = Nb/2
##########    i = 0
##########    for i in range(len(I)):
##########        _A.append( dot(transpose(I[i]),dot(_A[i],I[i])) )
##########
##########
##########    x = zeros(2*Nb,float64)
###########    x[0:Nb], x[Nb:2*Nb] = 1. * X, 1. * Y
##########
##########    clf()
##########    scatter(x[0:Nb],x[Nb:2*Nb],c='b')
##########    scatter(gaussx[0:Nb],gaussx[Nb:2*Nb],c='r')
##########    draw()
##########    raw_input("")
##########
##########
##########    for q in range(2):
##########        MultigridStep(_A, b, I, x)
##########
##########        clf()
##########        scatter(x[0:Nb],x[Nb:2*Nb],c='b')
##########        scatter(gaussx[0:Nb],gaussx[Nb:2*Nb],c='r')
##########        draw()
##########        raw_input("")
##########
##########
##########    x = gaussx
##########    X,Y = x[0:Nb] + X, x[Nb:2*Nb] + Y
##########
##########    Max = 0
##########    for i in range(Nb):
##########        _a = abs(X[i] - __X[i])
##########        _b = abs(Y[i] - __Y[i])
##########        _Max = (_a**2 + _b**2)**.5 / (__X[i]**2 + __Y[i]**2)**.5
##########
###########        print _Max
##########        Max = max(Max, _Max)
##########
##########    print "Max relative error:", Max
###########    raw_input("")


    # GET TIME
    t1 = time.clock()

    u, v = 1. * Old_u, 1. * Old_v
    IB_c.CentralDerivative_x (N, M, h, u, ux)
    IB_c.CentralDerivative_x (N, M, h, v, vx)
    IB_c.CentralDerivative_y (N, M, h, u, uy)
    IB_c.CentralDerivative_y (N, M, h, v, vy)

    # GET TIME
    t2 = time.clock()

    Xss, Yss = FiberForce (Nb, X, Y, Xss, Yss, Links, Tethers)   
    fx, fy = ForceToGrid (N, M, h, Nb, hb, Old_X, Old_Y, Xss, Yss)
    print "errr..", time.clock() - t2
    
    fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)
    fx += 1.*ones((N,M),float64)

    fx = fft2(fx)
    fy = fft2(fy)

    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.   

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)

    u = ifft2(u).real
    v = ifft2(v).real
    P = ifft2(P).real

    print "Second fluid solve", time.clock() - t1
    j, k, ___s, ___L = Links[5]
    print "                    ", X[j]

    count += 1
    if count % 1 == 0:
        clf()
        imshow(P)
        scatter(M*Y,N*X/2)
        quiver(M*Y,N*X/2,gaussx[Nb:2*Nb],-gaussx[0:Nb])
        draw()
#        print gaussx[0:Nb]
#        raw_input("")
