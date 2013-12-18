import sys
sys.path.append("c:\HeartValve2\IB_c\Release")
from scipy import *
from numpy import *
from numpy.fft import fft2, ifft2, fft, ifft, rfft
import IB_c
from pylab import *
from scipy import sparse

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


def InitMultigridIMatrices(Nb):
    I = []
    _N = Nb
    _N2 = 2 * _N
    while _N > 2:
        _M = ceil(double(_N) / 2.)
        _I = zeros((2*_N,2*_M), float64)
        for i in range(_M):
            if 2*i < _N:
                _I[2*i,i] = 1.
                _I[2*i+_N,i+_M] = 1.

            if 2*i + 1 < _N:
                _I[2*i+1,(i+1)%(_M)] = .5
                _I[2*i+1,(i)%(_M)] = .5
                _I[2*i+1+_N,(i+1)%(_M)+_M] = .5
                _I[2*i+1+_N,(i)%(_M)+_M] = .5

            

        _N = _M
        I.append(sparse.csc_matrix(_I))

    return I

def GaussSeidel (N, A, b, x, damp = .3):
    _damp = 1. - damp
    for i in range(N):
        S = dot(A[i,:], x) - A[i,i] * x[i]

        x[i] = _damp * x[i] + damp * (b[i] - S) / A[i,i]

def MultigridStep(A, b, I, x, damp = .3):
    if len(I) <= 2:
        return linalg.solve(A[0],b)
        return x

#    print A[0].shape, I[0].T.shape, b.shape, x.shape

#    r = dot(I[0].T, b - dot(A[0], x))
    r = I[0].T * (b - dot(A[0], x))
#    print "  ",r.shape, A[0].shape, x.shape, dot(A[0], x).shape
    _x = zeros(len(r), float64)
    _x = MultigridStep(A[1:], r, I[1:], _x, damp)
#    x += dot(I[0], _x)
    x += I[0] * _x
    VCycle(A, b, I, x, damp)

    return x
    
def VCycle(A, b, I, x, damp = .3):
    if len(I) <= 1:
        return

    GaussSeidel(len(x), A[0], b, x)
    
    r = I[0].T * (b - dot(A[0], x))
    _x = zeros(len(r), float64)
    VCycle(A[1:], r, I[1:], _x)
    VCycle(A[1:], r, I[1:], _x)    
    x += I[0] * _x

    GaussSeidel(len(x), A[0], b, x)


def InitVelField(_N, _M, _h, h, dt, rho = 1., mu = 1., DeltaType = 0):
    WideLambda = zeros((_N,_M),float64)
    ShortLambda = zeros((_N,_M),float64)
    IB_c.InitWideLaplacian(_N, _M, _h, WideLambda)
    IB_c.InitShortLaplacian(_N, _M, _h, ShortLambda)
    DxSymbol = InitDxSymbol(_N, _M, _h)
    DySymbol = InitDySymbol(_N, _M, _h)

    r = int(ceil(3. * h / _h))

    fx = zeros((_N,_M), float64)
    for j in range(-r, r + 1):
        deltx = Delta (h, j * _h, DeltaType)
        for k in range(-r, r + 1):
            delt = deltx * Delta (h, k * _h, DeltaType) * 1.
            fx[j%_N][k%_M] = fx[j%_N][k%_M] + delt
 #       print j%_N, k%_M, fx[j%_N][k%_M]


    fx, fy = fft2(dt * fx), zeros((_N,_M), float64)
    
    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy, rho, mu)
    u = 1.*ifft2(u).real
    v = 1.*ifft2(v).real
#    P = ifft2(P).real

    Fx1 = array(zeros((_N,_M), float64))
    Fy1 = array(zeros((_N,_M), float64))

    IB_c.WholeGridSpread(u, float(h), float(_h), int(r), Fx1, DeltaType)
    IB_c.WholeGridSpread(v, float(h), float(_h), int(r), Fy1, DeltaType)



    fy = zeros((_N,_M), float64)
    for j in range(-r, r + 1):
        deltx = Delta (h, j * _h, DeltaType)
        for k in range(-r, r + 1):
            delt = deltx * Delta (h, k * _h, DeltaType) * 1.
            fy[j%_N][k%_M] = fy[j%_N][k%_M] + delt
 #       print j%_N, k%_M, fx[j%_N][k%_M]

    fx, fy = zeros((_N,_M), float64), fft2(dt * fy)
    
    P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
    P[0,0] = 0.

    u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy, rho, mu)
    u = 1.*ifft2(u).real
    v = 1.*ifft2(v).real

    Fx2 = array(zeros((_N,_M), float64))
    Fy2 = array(zeros((_N,_M), float64))

    IB_c.WholeGridSpread(u, float(h), float(_h), int(r), Fx2, DeltaType)    
    IB_c.WholeGridSpread(v, float(h), float(_h), int(r), Fy2, DeltaType)    

    return Fx1, Fy1, Fx2, Fy2


def ComputeMatrix (X, Y, _N, _M, Nb, _h, hb, UField, VField, band = -1):
    if band == -1:
        band = Nb
        
    A = zeros((2*Nb,2*Nb),float32)
    B = zeros((2*Nb,2*Nb),float32)
    for i in range(Nb):
        for j in range(Nb):
            if abs(j - i) < band or abs(j - i) > Nb - band:
                a, b = (X[i] - X[j]) / _h, (Y[i] - Y[j]) / _h
                _i = floor(a)
                _j = floor(b)
                a, b = a - _i, b - _j

                if i >= j:
                    A[i,j] = (1.-a)*(1.-b) * hb * UField[(_i)%_N,(_j)%_M]
                    A[i+Nb,j+Nb] = (1.-a)*(1.-b) * hb * UField[(_j)%_N, (_i)%_M]

                    A[i,j] += (a)*(1.-b) * hb * UField[(_i+1)%_N,(_j)%_M]
                    A[i+Nb,j+Nb] += (a)*(1.-b) * hb * UField[(_j)%_N, (_i+1)%_M]
                
                    A[i,j] += (1.-a)*(b) * hb * UField[(_i)%_N,(_j+1)%_M]
                    A[i+Nb,j+Nb] += (1.-a)*(b) * hb * UField[(_j+1)%_N, (_i)%_M]

                    A[i,j] += (a)*(b) * hb * UField[(_i+1)%_N,(_j+1)%_M]
                    A[i+Nb,j+Nb] += (a)*(b) * hb * UField[(_j+1)%_N, (_i+1)%_M]

                    A[j,i] = A[i,j]
                    A[j+Nb,i+Nb] = A[i+Nb, j+Nb]

                A[i+Nb,j] = (1.-a)*(1.-b) * hb * VField[(_i)%_N,(_j)%_M]
                A[i+Nb,j] += (a)*(1.-b) * hb * VField[(_i+1)%_N,(_j)%_M]    
                A[i+Nb,j] += (1.-a)*(b) * hb * VField[(_i)%_N,(_j+1)%_M]
                A[i+Nb,j] += (a)*(b) * hb * VField[(_i+1)%_N,(_j+1)%_M]                            

                A[j,i+Nb] = A[i+Nb,j]

    return A


# Initializes the laplacian egeinvalue thingy
def InitWideLaplacian (N, M, h, Lambda):    
    for j in range(N):
        for k in range(M):
            Lambda[j,k] = -sin(2*pi*j / N)**2 / h**2 - sin(2*pi*k / M)**2 / h**2

            if Lambda[j,k] == 0:
                Lambda[j,k] = .01

    return Lambda

# Initializes the laplacian egeinvalue thingy
def InitShortLaplacian (N, M, h, Lambda):    
    for j in range(N):
        for k in range(N):
            Lambda[j,k] = (2 * cos(2.*pi*j/N) - 2.) / h**2 + (2 * cos(2.*pi*k/N) - 2.) / h**2

            if Lambda[j,k] == 0:
                Lambda[j,k] = .01

    return Lambda


def InitDxSymbol (N, M, h):
    _DxSymbol = zeros((N,M), float64)
    IB_c.InitDxSymbol(N,M,h,_DxSymbol)
    return 1j * _DxSymbol;


def InitDySymbol (N, M, h):
    _DySymbol = zeros((N,M), float64)
    IB_c.InitDySymbol(N,M,h,_DySymbol)
    return 1j * _DySymbol;


# Centered derivative for a periodic function
def CentralDerivativePeriodic (Nb, hb, X, Xs):
    for j in range(1,Nb-1):
        Xs[j] = (X[j+1] - X[j-1]) / (2. * hb)

    Xs[0] = (X[1] - X[Nb-1]) / (2. * hb)
    Xs[Nb-1] = (X[0] - X[Nb-2]) / (2. * hb)
    

# Second derivative for a periodic function
def SecondDerivativePeriodic (Nb, hb, X, Xss):
    for j in range(1,Nb-1):
        Xss[j] = (X[j+1] - 2 * X[j] + X[j-1]) / hb**2

    Xss[0] = (X[1] - 2 * X[0] + X[Nb-1]) / hb**2
    Xss[Nb-1] = (X[0] - 2 * X[Nb-1] + X[Nb-2]) / hb**2
        

# Central derivative in the x coordinate
def CentralDerivative_x (N, M, h, f, df):
    for j in range(M):
        for i in range(1,N-1):
            df[i][j] = (f[i+1][j] - f[i-1][j]) / (2. * h)

        df[0][j] = (f[1][j] - f[N-1][j]) / (2. * h)
        df[N-1][j] = (f[0][j] - f[N-2][j]) / (2. * h)


# Central derivative in the y coordinate
def CentralDerivative_y (N, M, h, f, df):
    for i in range(N):
        for j in range(1,M-1):
            df[i][j] = (f[i][j+1] - f[i][j-1]) / (2. * h)
        df[i][0] = (f[i][1] - f[i][M-1]) / (2. * h)
        df[i][M-1] = (f[i][0] - f[i][M-2]) / (2. * h)


# Combine boundary force, convection and previous velocity into explicit terms
def ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy, rho = 1.):
    return rho * u + dt * (fx - rho * (u * ux + v * uy)), rho * v + dt * (fy - rho * (u * vx + v * vy))

# Solve for the pressure in transform space
def Solve_P_Hat (dt, Lambda, DxSymbol, DySymbol, cx_Hat, cy_Hat):
    return (DxSymbol * cx_Hat + DySymbol * cy_Hat) / (dt * Lambda)

# Solve for the velocity in transform space
def Solve_uv_Hat (dt, Lambda, DxSymbol, DySymbol, P_Hat, cx_Hat, cy_Hat, rho = 1., mu = 1.):
    return (-dt * DxSymbol * P_Hat + cx_Hat) / (rho - dt * mu * Lambda), (-dt * DySymbol * P_Hat + cy_Hat) / (rho - dt * mu * Lambda)
  

# Delta approximation
def Delta (h, r, DeltaType = 0):
##    if abs(r) < 2 * h:
##        return (1. + cos(pi * r / (2*h))) / (4. * h)
##    else:
##        return 0

    if DeltaType == 0:
        if abs(r) < 2 * h:
            return (1. + cos(pi * r / (2*h))) / (4. * h)
        else:
            return 0
    else:
        x = r / h
        absx = abs(x)
        if absx <= 2:
            if absx <= 1:
                return .125 * (3. - 2 * absx + sqrt(1. + 4. * absx - 4. * x * x)) / h
            else:
                return .125 * (5. - 2 * absx - sqrt(-7. + 12. * absx - 4 * x * x)) / h
        else:
            return 0

# Extend the fiber force distribution onto the grid
def ForceToGrid (N, M, h, Nb, hb, X, Y, Boundary_fx, Boundary_fy, DeltaType = 0):
    fx = zeros((N,M), float64)
    fy = zeros((N,M), float64)
 
##    # Loop through fiber points
##    for i in range(Nb):
##        # Modify force for nearby points
##        jMin = int(X[i] / h - 2.)
##        jMax = jMin + 5
##        kMin = int(Y[i] / h - 2.)
##        kMax = kMin + 5
##       
##        for j in range(jMin, jMax + 1):
##            deltx = Delta (h, j * h - X[i])
##            for k in range(kMin, kMax + 1):
##                delt = deltx * Delta (h, k * h - Y[i]) * hb
##                fx[j%N][k%M] = fx[j%N][k%M] + Boundary_fx[i] * delt
##                fy[j%N][k%M] = fy[j%N][k%M] + Boundary_fy[i] * delt

    IB_c.ForceToGrid (N, M, h, Nb, hb, X, Y, Boundary_fx, Boundary_fy, fx, fy, DeltaType)

    return fx, fy


# Calculate the fiber velocity based on the surrounding fluid velocity
def ToFiber (N, M, h, Nb, hb, X, Y, f, F, DeltaType = 0):
    for j in range(Nb):
        F[j] = 0.

    # Loop through fiber points
    for i in range(Nb):
        # Modify velocity for nearby points
        jMin = int(X[i] / h - 2.)
        jMax = jMin + 5
        kMin = int(Y[i] / h - 2.)
        kMax = kMin + 5
        
        for j in range(jMin, jMax + 1):
            deltx = Delta (h, j%N * h - X[i], DeltaType)
            for k in range(kMin, kMax + 1):
                delt = deltx * Delta (h, k%M * h - Y[i], DeltaType) * h**2
                F[i] += f[j][k] * delt

    
def VelToFiber (N, M, h, Nb, hb, dt, X, Y, u, v, DeltaType = 0):
##    Xvel = zeros(Nb, float64)
##    Yvel = zeros(Nb, float64)
##    
##    # Loop through fiber points
##    for i in range(Nb):
##        # Modify velocity for nearby points
##        jMin = int(X[i] / h - 2.)
##        jMax = jMin + 5
##        kMin = int(Y[i] / h - 2.)
##        kMax = kMin + 5
##        
##        for j in range(jMin, jMax + 1):
##            deltx = Delta (h, j * h - X[i])
##            for k in range(kMin, kMax + 1):
##                delt = deltx * Delta (h, k * h - Y[i]) * h**2
##                Xvel[i] += u[j%N][k%M] * delt
##                Yvel[i] += v[j%N][k%M] * delt

    Xvel = zeros(Nb, float64)
    Yvel = zeros(Nb, float64)
    IB_c.VelToFiber (N, M, h, Nb, hb, dt, X, Y, u, v, Xvel, Yvel, DeltaType)

    return dt * Xvel, dt * Yvel

# Ellipse initial condition
def Ellipse (Nb, X, Y, a, b, cx, cy):
    incr = 2. * pi / Nb
    for i in range(Nb):
        X[i] = a * cos(incr * i) + cx
        Y[i] = b * sin(incr * i) + cy

def CrazyEllipse (Nb, X, Y, a, b, cx, cy, r, p):
    incr = 2. * pi / Nb
    _incr = incr * p
    for i in range(Nb):
        X[i] = a * cos(incr * i) + cx + r * cos(_incr * i)
        Y[i] = b * sin(incr * i) + cy + r * sin(_incr * i)
    
