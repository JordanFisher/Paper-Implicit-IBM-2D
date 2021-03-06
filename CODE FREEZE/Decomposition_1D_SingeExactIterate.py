import sys
from scipy.linalg import eig as Eig
import scipy
from pylab import *
from numpy import *
import time
import math
show()

  
N = 64


NumTerms = 1
z = ones(2 * N * NumTerms)
f = ones(N * N)
F = zeros((N, N))
r = ones(N * N)

h = 1. / N
for j in range(N):
    for k in range(N):
##        f[j + N * k] = j * k + cos(j)**2 + sin(k)**2 + sin(j * k)**2 + j**2

        f[j + N * k] = cos(j**2) * cos(k) + j
##        f[j + N * k] = 1.
##        f[j + N * k] = 1. / ((j - k)**2 + 1)

f /= dot(f, f)**.5

for j in range(N):
    for k in range(N):
        F[j, k] = f[j + N * k]

for j in range(N):
    for l in range(NumTerms):
        z[j + l * 2 * N] = cos(j * l + j + l) + 0
        z[j + N + l * 2 * N] = sin(j * l + j + l) + 0


def EvalDecomp(z):
    val = zeros(N * N, float64)

    for j in range(N):
        for k in range(N):
            for l in range(NumTerms):
                val[j + k * N] += z[j + 2 * N * l] * z[k + N + 2 * N * l]

    return val

def EvalDecompToPlane(z):
    val = zeros((N, N), float64)

    for j in range(N):
        for k in range(N):
            for l in range(NumTerms):
                val[j, k] += z[j + 2 * N * l] * z[k + N + 2 * N * l]

    return val
            

VarsToSkipLeft = [2, 1, 3, 0, 0, 0, 0]
VarsToSkipRight = [1, 2, 0, 0, 0, 0, 0]
VarsToSkip = [3, 3, 3, 0, 0, 0, 0]
TotalSkips = 9

def Jacobian(z):
##    J = zeros((N * N, (2 * N - 1) * NumTerms), float64)
    J = zeros((N * N, 2 * N * NumTerms - TotalSkips), float64)    
    _J = zeros((N * N, 2 * N), float64)    

    n = 0

    for l in range(NumTerms):
        for j in range(N):
            for k in range(N):            
                _J[j + k * N, j] = z[k + N + 2 * N * l]
                _J[j + k * N, k + N] = z[j + 2 * N * l]
        J[:, n : n + 2 * N - VarsToSkip[l]] = _J[:, VarsToSkipLeft[l] : 2 * N - VarsToSkipRight[l]]
        n += 2 * N - VarsToSkip[l]

    return J

def Iterate(z, f):
    e = EvalDecomp(z)
    r = f - e
    cur_error = abs(r).max()
    cur_z = 1. * z
    print cur_error

    J = Jacobian(z)
    
    delta = linalg.solve(dot(J.T, J), dot(J.T, r))

    holdr = 1. * r

    n, m = 0, 0
    for l in range(NumTerms):
        n = 2 * N * l
        z[n + VarsToSkipLeft[l] : n + 2 * N - VarsToSkipRight[l]] += delta[m : m + 2 * N - VarsToSkip[l]]
        m += 2 * N - VarsToSkip[l]


    e = EvalDecomp(z)
    r = f - e
    error = abs(r).max()

    return z

    while error > cur_error and abs(error) > .000000001:
        z = (z + cur_z) / 2

        e = EvalDecomp(z)
        r = f - e
        error = abs(r).max()

        print " ", error

    return z#, delta, holdr

def SingularCount(z):
    J = Jacobian(z)
    a, b = eig(dot(J.T,J))

    count = 0

    for i in range(2 * N * NumTerms - TotalSkips):
            if abs(a[i]) < .0000001:
                    count += 1

    print count

def ExactSolveMatrix(F):
    M = identity(2 * N, float64)

    M[:N, N:] = -F
    M[N:, :N] = -F.T

    return M

def ExactSolveMatrix(F, z):
    M = identity(2 * N, float64)

    M[:N, N:] = -F
    M[N:, :N] = -F.T

    M[:N, :N] *= dot(z[N:], z[N:])
    M[N:, N:] *= dot(z[:N], z[:N])

    return M

##z2 = 1 * z.T
z = 1 * ones(2*N)
z2 = 1 * ones(2*N)
for i in range(500):
    w = EvalDecompToPlane(z)
    z2[:N] = dot((F-w), z2[N:]) / dot(z2[N:], z2[N:])
    z2[N:] = dot((F-w).T, z2[:N]) / dot(z2[:N], z2[:N])

    w2 = EvalDecompToPlane(z2)
    print abs(w + w2 - F).max(),

    w2 = EvalDecompToPlane(z2)
    z[:N] = dot((F-w2), z[N:]) / dot(z[N:], z[N:])
    z[N:] = dot((F-w2).T, z[:N]) / dot(z[:N], z[:N])

    w = EvalDecompToPlane(z)
    print abs(w + w2 - F).max()

