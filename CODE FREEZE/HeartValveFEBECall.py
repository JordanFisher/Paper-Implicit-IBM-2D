import sys
sys.path.append("c:\HeartValve")
sys.path.append("c:\HeartValve\IB_c\Release")
from pylab import *
from numpy import *
from numpy.fft import fft2, ifft2, fft, ifft, rfft
from IB_Methods import *
import IB_c
from ValveSetup import *
import time
#show()

def FastForce(X, Y, Xss, Yss, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData):
    Nb = len(X)
    
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
    
 


  

def ExplicitHeartValveSim(N, dx, dt, T):
    Domain_x, Domain_y = 2., 1.
    #h = .003#.007
    h = Domain_x / N
    dx = h / 2.
    N, M = int(Domain_x / h), int(Domain_y / h)
    #dt = .0000065
    CurT = 0

    _s = 1000000000. / dx**2
    #_s = 500000000000. / dx**2
    dt = 30. * h / _s**.5
    Current = 100.


    WideLambda = zeros((N,M),float64)
    ShortLambda = zeros((N,M),float64)
    IB_c.InitWideLaplacian(N, M, h, WideLambda)
    IB_c.InitShortLaplacian(N, M, h, ShortLambda)
    DxSymbol = InitDxSymbol(N, M, h)
    DySymbol = InitDySymbol(N, M, h)

    #UField, VField = InitVelField(N, M, h, h, dt)

    X, Y, Links, Tethers, Parts = ConstructValve (_s, 0., Domain_x, 0., Domain_y, dx, False,1.3,1.1)#.01)
    #X, Y, Links, Tethers, Parts = ConstructValve (_s, 0., Domain_x, 0., Domain_y, .012, True,1.5,1.)#.01)
    #X, Y, Links, Tethers, Valve = ConstructValve (_s, 0., 2., 0., 1., .03)#.01)
    Links = []
    Nb = len(X)
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


    PlotValve (X, Y, Links)


    Times = []
    A = zeros((2*Nb,2*Nb), float64)

    print N, dx, Nb, dt

    b = zeros(2*Nb,float64)
    count = 0
    TotalTime = 0
    print
    print "Time:",
    while CurT < T:
        Old_X, Old_Y = 1. * X, 1. * Y
#        print "Time:", CurT, count
#        print CurT,
#        CurT += dt

        Time = time.time()

        IB_c.CentralDerivative_x (N, M, h, u, ux)
        IB_c.CentralDerivative_x (N, M, h, v, vx)
        IB_c.CentralDerivative_y (N, M, h, u, uy)
        IB_c.CentralDerivative_y (N, M, h, v, vy)

        FastForce(X, Y, Xss, Yss, LinkIndices, LinkData, LinkCurL, TetherIndex, TetherData)
#        Xss, Yss = FiberForce (Nb, X, Y, Xss, Yss, Links, Tethers)   
        fx, fy = ForceToGrid (N, M, h, Nb, hb, X, Y, Xss, Yss)
        
        fx, fy = ExplicitTerms (dt, u, v, ux, uy, vx, vy, fx, fy)
        fx += dt * Current * ones((N,M),float64)
        
        fx = fft2(fx)
        fy = fft2(fy)
        
        P = Solve_P_Hat (dt, WideLambda, DxSymbol, DySymbol, fx, fy)
        P[0,0] = 0.   

        u, v = Solve_uv_Hat (dt, ShortLambda, DxSymbol, DySymbol, P, fx, fy)

        u = ifft2(u).real
        v = ifft2(v).real

        Xvel, Yvel = VelToFiber (N, M, h, Nb, hb, 1., X, Y, u, v)
        X, Y = X + dt * Xvel, Y + dt * Yvel
        
#        print "Explicit time:", time.time() - Time
#        Times.append(time.time() - Time)
#        print "avg:", sum(Times)/len(Times)

##        if time.time()-Time > 1:
##            P = ifft2(P).real
##            clf()
##            imshow(P)
##            scatter(M*Y,N*X/2)
##            quiver(M*Y,N*X/2,Yvel,-Xvel)
##            print "Halt!"
##            raw_input("")



        Time = time.time() - Time
        TotalTime += Time
        count += 1

        Ts = [.0001,.0002,.0005,.001,.005]
        for _T in Ts:
            if CurT < _T and CurT + dt >= _T:
                print "Time", _T, ":", count, TotalTime, TotalTime / count
        #print "Time:", CurT
        CurT += dt


        if count % 500 == 0:
            print CurT, count
            P = ifft2(P).real
            clf()
            #imshow(P)
            imshow(u)
            colorbar()
            
            scatter(M*Y,N*X/2)
            quiver(M*Y,N*X/2,Yvel,-Xvel)
    #        #_!_print gaussx[0:Nb]
    #        raw_input("")

    print
    print
    print "End:", count, TotalTime, TotalTime / count
    return Times

Data = [(128, .009, .0000001, .00001),
        (256, .0045,.000003, .000001),
        (384, .003, .000003, .0000001),
        (512,.00225,.000003, .00000001)]        

#Data = [(250, .009, .000001, .0001)]
Time = []
for d in Data:
    N, dx, dt, T = d
    Time.append(ExplicitHeartValveSim(N, dx, dt, T))
