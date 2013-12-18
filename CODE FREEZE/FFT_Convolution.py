import sys
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

root = 'c:\HeartValve2\\'
Terms = 20

def Shift(f, x, y):
    w = f * 0
    w[x:, y:] = f[:-x, :-y]
    w[x:, :y] = f[:-x, -y:]
    w[:x, y:] = f[-x:, :-y]
    w[:x, :y] = f[-x:, -y:]
    return w
raise

class Decomposition:
    def __init__(self, file_root, N, Type, U_Width, Terms):
        name = file_root + 'Decomposition\\N_' + str(N) + '\\' + Type + '\\Panel_Width_' + str(U_Width)

        self.Terms = Terms
        self.U_Width = U_Width

        self.U, self.V = [], []
        for l in range(Terms):
            self.U.append(numpy.load(name + '\\U\\' +  str(l) + '.npy'))
            self.V.append(numpy.load(name + '\\V\\' +  str(l) + '.npy'))            

class AllDecompositions:
    def __init__(self, file_root, N, Terms):
        self.UField = [None, None]

        U_Width = N / 4
        while U_Width > 4:
            self.UField.append(Decomposition(file_root, N, 'UField', U_Width, Terms))
            U_Width /= 2

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


  
N = 256

All = AllDecompositions(root, N, Terms)
MaxDepth = len(All.UField)




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

##UField = ones((N,N),float64)

##UField = zeros((N,N), float64)
##for j in range(N):
##    for k in range(N):
##        UField[j,k] = cos(2*j*pi*h)


print UField[0,0]

Nb = 2 * N
hb = 1. / Nb
_X = zeros(Nb, float64)
_Y = zeros(Nb, float64)
Ellipse (Nb, _X, _Y, 1/3., 1/4., .5, .5)



X = zeros((Nb,2),float64)
X[:,0], X[:,1] = _X, _Y
##X = floor(X*N) / N


A = zeros((2*Nb,2*Nb),float64)
IB_c.ComputeMatrix (X[:,0], X[:,1], _N, _N, Nb, _h, hb, UField, VField, UField2, VField2, A, -1)
##IB_c.ComputeMatrix (_X, _Y, _N, _N, Nb, _h, hb, UField, VField, UField2, VField2, A, -1)

## tree
def PointIsInBox(Point, BL, TR):
    BL, TR, Point = [0,0], TR - BL, Point - BL
    Point[0] -= math.floor(Point[0])
    if (Point[0] < BL[0]): return False
    if (Point[0] >= TR[0]): return False
    Point[1] -= math.floor(Point[1])    
    if (Point[1] < BL[1]): return False    
    if (Point[1] >= TR[1]): return False    
    return True

def Lerp(Field, x):
##    x = abs(x)
    i00, j00 = int(x[0]), int(x[1])
    i10, j10 = i00 + 1, j00
    i01, j01 = i00, j00 + 1
    i11, j11 = i10, j01

##    print Field[i00, j00], Field[i01, j01], Field[i10, j10], Field[i11, j11]
    return Field[i00, j00] * (i11 - x[0]) * (j11 - x[1]) + \
           Field[i01, j01] * (i11 - x[0]) * (x[1] - j00) + \
           Field[i10, j10] * (x[0] - i00) * (j11 - x[1]) + \
           Field[i11, j11] * (x[0] - i00) * (x[1] - j00)


MinPointsToHaveChildren = 10
class Panel:
    def __init__(self, Parent, X, BL, TR, Level):
        self.Level = Level
        self.BL, self.TR = BL, TR
        self.center = (TR + BL) / 2.
        self.Size = self.TR - self.BL
        self.WellSeparated_BL, self.WellSeparated_TR = self.BL - self.Size, self.TR + self.Size
                    
        if (Parent == 0):
            self.N = X.shape[0]
            self.Index = array(range(self.N))
        else:
            self.N = 0
            self.Index = zeros(Parent.N, int32)

            for i in range(Parent.N):
                # Check to see if i-th point in the parent panel is in this panel
                x = X[Parent.Index[i]]
                if (PointIsInBox(x, self.BL, self.TR)):
                    self.Index[self.N] = Parent.Index[i]
                    self.N += 1

        # If we have enough points create children panels
        self.Children = []
        if self.N > MinPointsToHaveChildren and Level + 1 < MaxDepth:
##        if self.N > 0 and Level + 1 < MaxDepth:
            self.Childless = False
            center = self.center
            self.Children.append(Panel(self, X, BL, center, Level + 1))
            self.Children.append(Panel(self, X, center, TR, Level + 1))
            self.Children.append(Panel(self, X, array([center[0], BL[1]]), array([TR[0], center[1]]), Level + 1))
            self.Children.append(Panel(self, X, array([BL[0], center[1]]), array([center[0], TR[1]]), Level + 1))
        else:
            self.Childless = True
            
    def CalcUV(self, Terms, Lookup):
        self.Terms = Terms

        if Lookup.UField[self.Level] == None:
            self.UField_FarField = None
            self.UField_U = None
        else:
            self.UField_FarField = zeros(self.Terms, float64)
            self.UField_U = zeros((self.N, self.Terms), float64)
        
            for i in range(self.N):
                dif = (X[self.Index[i]] - self.center) * N #+ [N/2, N/2]
                
                for l in range(self.Terms):
                    self.UField_U[i, l] = Lerp(Lookup.UField[self.Level].U[l], dif)
##                    print self.UField_U[i, l], Lookup.UField[self.Level].U[l][int(dif[0]), int(dif[1])]

        for child in self.Children:
            child.CalcUV(Terms, Lookup)


    def CalcSeriesTerms(self, F):
        if self.UField_U != None:
            self.UField_FarField *= 0
            
            for i in range(self.N):
                f = F[self.Index[i]]
                for l in range(self.Terms):
                    self.UField_FarField[l] += f * self.UField_U[i, l]

##                if f > 0:
##                    print self.Index[i], i, self.N, self.BL, self.TR

##        print self.UField_FarField
        for child in self.Children:
            child.CalcSeriesTerms(F)
##        print self.UField_FarField
        
    def GetBoundary(self, Depth, Lines = None):
        if Lines == None:
            return self.GetBoundary(Depth, ([], [], [], []))
        else:    
            x1, x2, y1, y2 = Lines

            BL = self.BL
            TR = self.TR
            BR = [self.TR[0], self.BL[1]]
            TL = [self.BL[0], self.TR[1]]

            x1.extend([BL[0], BR[0], TR[0], TL[0]])
            x2.extend([BR[0], TR[0], TL[0], BL[0]])
            y1.extend([BL[1], BR[1], TR[1], TL[1]])
            y2.extend([BR[1], TR[1], TL[1], BL[1]])

            if Depth != 0:
                for child in self.Children:
                    child.GetBoundary(Depth - 1, (x1, x2, y1, y2))

            return x1, x2, y1, y2

    def DrawBoundary(self, Depth):
        x1, x2, y1, y2 = self.GetBoundary(Depth)
        plot([x1, x2], [y1, y2], c='k', linewidth=2)

    def DrawPoints(self, X):
        if self.N > 0:
            scatter(X[self.Index[:self.N],0], X[self.Index[:self.N],1])

Count = 0
def EvalPoint(x, P, Lookup, X, F):
    Sum = 0.

    if P.N == 0:
        return Sum

    # Check to see if we are well seperated from the panel P
    if not PointIsInBox(x, P.WellSeparated_BL, P.WellSeparated_TR):
        dif = (x - P.center) * N
        
        for l in range(P.Terms):
            Sum += P.UField_FarField[l] * Lerp(Lookup.UField[P.Level].V[l], dif)

    else:
        # Check to see if P has children
        if not P.Childless:
            for child in P.Children:
                Sum += EvalPoint(x, child, Lookup, X, F)
        else:
            # Do a direct summation
            for i in range(P.N):
                y = X[P.Index[i]]
                Sum += Lerp(UField, (x - y) * N) * F[P.Index[i]]

    return Sum

Offset = 0
BoolOffset = 0
def FastEvalPoint(j, x, P, Lookup, X, F):
    global Offset
    global BoolOffset
    Sum = 0.

    if P.N == 0:
        return Sum
    
    # Check to see if we are well seperated from the panel P
    WellSeparated = PointInfoBool[j, BoolOffset]
    BoolOffset += 1
    if WellSeparated:
        dif = (x - P.center) * N
        
        for l in range(P.Terms):
            Sum += P.UField_FarField[l] * PointInfo[j, Offset]
            Offset += 1
    else:
        # Check to see if P has children
        if not P.Childless:
            for child in P.Children:
                Return = FastEvalPoint(j, x, child, Lookup, X, F)
                Sum += Return
        else:
            # Do a direct summation
            for i in range(P.N):
                Sum += PointInfo[j, Offset] * F[P.Index[i]]
                Offset += 1

    return Sum

def DirectEvalPoint(Index, X, F):
    Sum = 0
    for i in range(Nb):
        #Sum += Lerp(UField, (x - X[i]) * N) * F[i]
        #Sum += Lerp(UField, (X[Index] - X[i]) * N) * F[i]
        Sum += A[Index,i] * F[i]
    return Sum


PointInfo = zeros((Nb, 2 * 12 * (MaxDepth - 2) * Terms))
PointInfoBool = zeros((Nb, 12 * (MaxDepth - 2)), numpy.bool)
def CalcPointInfo(j, P, Lookup, UField, X):
    global Offset
    global BoolOffset
    
    if P.N == 0:
        return
    
    # Check to see if we are well seperated from the panel P
    NotInBox = PointInfoBool[j, BoolOffset] = not PointIsInBox(X[j], P.WellSeparated_BL, P.WellSeparated_TR)
    BoolOffset += 1
    
    #if not PointIsInBox(X[j], P.WellSeparated_BL, P.WellSeparated_TR):
    if NotInBox:
        dif = (X[j] - P.center) * N
        
        for l in range(P.Terms):
            PointInfo[j, Offset] = Lerp(Lookup.UField[P.Level].V[l], dif)
##            print j, Offset, PointInfo[j, Offset]            
            Offset += 1
    else:
        # Check to see if P has children
        if not P.Childless:
            for child in P.Children:
                CalcPointInfo(j, child, Lookup, UField, X)
        else:
            # Do a direct summation
            for i in range(P.N):
                y = X[P.Index[i]]
                PointInfo[j, Offset] = Lerp(UField, (X[j] - y) * N)
##                print j, Offset, PointInfo[j, Offset]                
                Offset += 1


def CalcAllPointInfo(P, Lookup, UField, X):
    for i in range(Nb):
        CalcPointInfo(i, P, Lookup, UField, X)

Root = Panel(0, X, array([0,0]), array([1,1]), 0)

def GetField(U, V, x, y):
    w = UField * 0
    for l in range(len(U)):
        w += U[l] * V[l][x,y]
    return w

def GetUField(UField, x, y):
    w = UField * 0
    w[x:, y:] = UField[:-x, :-y]
    w[x:, :y] = UField[:-x, -y:]
    w[:x, y:] = UField[-x:, :-y]
    w[:x, :y] = UField[-x:, -y:]
    return w

#UField = GetUField(UField, N/2, N/2)

F = ones(Nb,float64)
#F *= 0
#F[0] = 1
x_index = Nb / 2 + 5
x = X[x_index]
Root.CalcUV(Terms, All)
Root.CalcSeriesTerms(F)

print
print EvalPoint(x, Root, All, X, F)
print DirectEvalPoint(x_index, X, F) / hb
##CalcAllPointInfo(Root, All, UField, X)
Offset = BoolOffset = 0
CalcPointInfo(Nb/2+5, Root, All, UField, X)
Offset = BoolOffset = 0
print FastEvalPoint(Nb/2+5, x, Root, All, X, F)

print
print


Reps = 30
t = time.time()
for i in range(Reps):
    Offset = BoolOffset = 0
    FastEvalPoint(Nb/2+5, x, Root, All, X, F)
print (time.time() - t) / Reps

Reps = 30
t = time.time()
for i in range(Reps):
    EvalPoint(x, Root, All, X, F)
print (time.time() - t) / Reps

t = time.time()
for i in range(Reps):
    DirectEvalPoint(x_index, X, F)
print (time.time() - t) / Reps


##
##
##t = time.time()
##for i in range(Reps):
##    Root.CalcSeriesTerms(F)
##print (time.time() - t) / (Reps * Nb)
