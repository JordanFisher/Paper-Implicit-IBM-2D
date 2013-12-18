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

N = 160
h = 1. / N

x, y = 0,0#.1*h, .25*h
sx, sy, = .2*h, .3*h

S = 0.
for j1 in range(-5,5):
    for k1 in range(-5,5):
        for j2 in range(-5,5):
            for k2 in range(-5,5):
                S += h**4*abs(Delta(h,j1*h)*Delta(h,k1*h) * Delta(h,j2*h-x)*Delta(h,k2*h-y) - Delta(h,j1*h-sx)*Delta(h,k1*h-sy) * Delta(h,j2*h-x-sx)*Delta(h,k2*h-y-sy))

print S
