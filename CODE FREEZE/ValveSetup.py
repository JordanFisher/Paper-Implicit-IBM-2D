from pylab import *
from numpy import *
from copy import copy

def ConstructValve (s, A, B, C, D, dx = .01, mid = True, HingeScale = 1., HingeDensity = 1., FixedHinge = False):
    X = []
    Links = []
    Tethers = []
    hr = 0#dx

    LeftSide, RightSide, Middle = [], [], []
    Parts = []

    HS = (B - A) / 128.
    LA = 2 * HS
    LB = .5 * HS
    LC = 8 * HS
    RH = .5 * HS

    XR = (B - A) / 6. + A
    RC = .92 * (D - C) / 5.04


    # Channel wall left of cushion
    _Part = []
    _x, _y = A + hr, C + hr
    _N = int((XR - RC - _x) / dx)
    ds = (XR - RC - _x) / _N

    for j in range(_N+1):
        X.append( (_x + j * ds, _y) )
        _Part.append(len(X)-1)

    # Lower cushion
    #   Radius is .5*RC, quarter circumference is .5 * pi * r
    r = .5 * RC
    _N = int(.5 * pi * r / dx)
    da = .5 * pi / _N

    q = len(X)
    for j in range(_N-1,-1,-1): X.append( (XR - RC + r * cos(j * da), _y + r - r * sin(j * da)) )    
    for j in range(_N-1,0,-1): X.append( (XR + r * cos(.5 * pi + j * da), _y + r + r * sin(.5 * pi + j * da)) )
    X.append( (XR, _y + RC) )    
    for j in range(1,_N): X.append( (XR + r * cos(.5 * pi - j * da), _y + r + r * sin(.5 * pi - j * da)) )    
    for j in range(0,_N): X.append( (XR + RC - r * cos(j * da), _y + r - r * sin(j * da)) )
    _Part = _Part + range(q,len(X))

    # Channel wall right of cushion
    _x = XR + RC
    _N = int((B - hr - _x) / dx)
    ds = (B - hr - _x) / _N

    for j in range(_N-1):
        X.append( (_x + j * ds, _y) )
        _Part.append(len(X)-1)
    Parts.append(_Part)



    # Upper cushion
    _N = int(.5 * pi * r / dx)

    q = len(X)
    for j in range(_N-2,-1,-1): X.append( (XR - RC - r * cos(pi - j * da), D - dx - r + r * sin(pi - j * da)) )
    for j in range(_N-1,0,-1): X.append( (XR + r * cos(.5 * pi + j * da), D - dx - r - r * sin(.5 * pi - j * da)) )
    X.append( (XR, D - RC - dx) ) 
    for j in range(1,_N): X.append( (XR + r * cos(.5 * pi - j * da), D - dx - r - r * sin(.5 * pi - j * da)) )    
    for j in range(0,_N-1): X.append( (XR + RC + r * cos(pi - j * da), D - dx - r + r * sin(pi - j * da)) )
    Parts.append(range(q,len(X)))

    # Add tethers
    for j in range(len(X)):
        Tethers.append( (j, X[j], s) )


    # Hinged valve
    y1 = C + (B - A) / 32. + RC     # Lowest point of valve
    y7 = D - (B - A) / 32. - RC     # Highest point of valve
    y4 = y1 + .3 * (y7 - y1)        # Center, point of rotation
    y2 = y1 + LA        # Center of lower semicircle cap
    y6 = y7 - LA        # Center of top semicircle cap
    y3 = y4 - LC        # Bottom of cavity
    y5 = y4 + LC        # Top of cavity

    #   Straight side components, below cavity
    _N = max(0, int((y3 - y2) / dx))
    if _N > 0:
        ds = (y3 - y2) / _N

    N1 = len(X)
    for j in range(_N+1):
        if mid or j == 0 or j == _N:
            X.append( (XR, y2 + j * ds) )
            Middle.append(len(X)-1)
        X.append( (XR - LA, y2 + j * ds) )
        LeftSide.append(len(X)-1)
        X.append( (XR + LA, y2 + j * ds) )
        RightSide.append(len(X)-1)
        
    #   Cavities
    RA = .5 * ((LA - LB)**2 + (y4 - y3)**2) / (LA - LB)
    XA = XR - LB - RA
    YA = y4

    theta = arcsin((YA - y3) / RA)
    _N = int(2 * theta * RA / dx)
    da = 2 * theta / _N

    for j in range(1, _N):
        _theta = j * da - theta
        if mid:# or j == 0 or j == _N:
            X.append( (XR, YA + RA * sin(_theta)) )
            Middle.append(len(X)-1)
            if j == _N/2:
                TetherHinge = len(X)-1
        X.append( (XA + RA * cos(_theta), YA + RA * sin(_theta)) )
        LeftSide.append(len(X)-1)        
        X.append( (2*XR - XA - RA * cos(_theta), YA + RA * sin(_theta)) )
        RightSide.append(len(X)-1)                

    #   Straight side components, above cavity
    _N = max(0, int((y6 - y5) / dx))
    if _N > 0:
        ds = (y6 - y5) / _N

    for j in range(_N+1):
        if mid or j == _N:        
            X.append( (XR, y5 + j * ds) )
            Middle.append(len(X)-1)
        X.append( (XR - LA, y5 + j * ds) )
        LeftSide.append(len(X)-1)        
        X.append( (XR + LA, y5 + j * ds) )
        RightSide.append(len(X)-1)
        
    #   Link together nodes in vertical column
    if mid:
        for j in range(N1, len(X)-3, 3):
            Links.append( (j, j+1) )
            Links.append( (j, j+2) )
            Links.append( (j, j+3) )
            Links.append( (j, j+4) )
            Links.append( (j, j+5) )
            Links.append( (j+1, j+4) )
            Links.append( (j+1, j+3) )
            Links.append( (j+2, j+5) )
            Links.append( (j+2, j+3) )
        N2 = len(X)-3
        Links.append( (N2, N2+1) )
        Links.append( (N2, N2+2) )
    else:
        for j in range(N1+4, len(X)-4, 2):
            Links.append( (j, j+1) )
            Links.append( (j, j+2) )
            Links.append( (j+1, j+2) )
            Links.append( (j-1, j+1) )
            Links.append( (j-1, j+2) )

        N2 = len(X)-3
        
#        Links.append( (N1+3, N1+5) )
        Links.append( (N1+3, N1+4) )

        Links.append( (N1+3, N1) )
        Links.append( (N1+3, N1+1) )
        Links.append( (N1, N1+1) )

        Links.append( (N1+4, N1+2) )
        Links.append( (N1+4, N1) )
        Links.append( (N1, N1+2) )

        Links.append( (N2-2, N2+1) )
        Links.append( (N2-2, N2) )
        Links.append( (N2, N2+1) )

        Links.append( (N2-1, N2) )
        Links.append( (N2-1, N2+2) )
        Links.append( (N2, N2+2) )
        
##        Links.append( (N2, N2+1) )
##        Links.append( (N2, N2+2) )

        

    Parts.append(copy(RightSide))
    Parts.append(copy(LeftSide))
    Parts.append(copy(Middle))

    #   Circular caps
    _N = int(pi * LA / dx)
    da = pi / _N

    N3 = len(X)
    _Part = []
    for j in range(1,_N):#+1):
        X.append( (XR + LA * cos(pi + j * da), y2 + LA * sin(pi + j * da)) )
        LeftSide.insert(0,len(X)-1)
        _Part.append(len(X)-1)
        Links.append( (len(X)-1, N1) )
    Parts.append(_Part)

    for j in range(N3,len(X)-1):
        Links.append( (j, j+1) )
    Links.append( (len(X)-1, N1+2) )
    Links.append( (N3, N1+1) )

    N4 = len(X)
    _Part = []
    for j in range(1,_N):#+1):
        X.append( (XR + LA * cos(pi - j * da), y6 + LA * sin(pi - j * da)) )
        LeftSide.append(len(X)-1)
        _Part.append(len(X)-1)
        Links.append( (len(X)-1, N2) )
    Parts.append(_Part)

    for j in range(N4,len(X)-1):
        Links.append( (j, j+1) )
    Links.append( (len(X)-1, N2+2) )
    Links.append( (N4, N2+1) )


    # Hinges
    if not FixedHinge:
        theta = pi / 4.
        x = XR - 1.4 * LA
        y = y4 + LA

        _N = 2*int((2. * pi * HingeScale * HingeDensity * RH) / dx)
        da = 2. * pi / _N

        _Part = []
        for j in range(_N):
            X.append( (x + HingeScale * RH * cos(j * da), y + HingeScale * RH * sin(j * da)) )
            _Part.append(len(X)-1)
            i = len(X) - 1
            Tethers.append( (i, X[i], s) )
        Parts.append(_Part)

        x = XR + 1.4 * LA
        y = y4 - LA

        _Part = []
        for j in range(_N):
            X.append( (x + HingeScale * RH * cos(j * da), y + HingeScale * RH * sin(j * da)) )
            _Part.append(len(X)-1)
            i = len(X) - 1
            Tethers.append( (i, X[i], s) )
        Parts.append(_Part)
    else:
        Tethers.append( (TetherHinge, X[TetherHinge], 10.*s) )
        Parts.append([TetherHinge])
        Parts.append([])

    x = zeros(len(X), float64)
    y = zeros(len(X), float64)

    for i in range(len(X)):
        x[i], y[i] = X[i]

    for i in range(len(Links)):
        j, k = Links[i]
        l = ((x[j] - x[k])**2 + (y[j] - y[k])**2)**.5
        Links[i] = j, k, s, l

#    print Parts
#    for q in Parts: print len(q)
    Parts[3].reverse()
    Parts[6].reverse()
    Parts.append(Parts[2] + Parts[6] + Parts[3] + Parts[5])
    Parts[3].reverse()
    Parts[6].reverse()
    return x, y, Links, Tethers, Parts

def PlotValve(x, y, Links):        
    __x1, __y1, __x2, __y2 = [], [], [], []
    for l in Links:
        i, j, s, L = l
        __x1.append (x[i])
        __y1.append (y[i])
        __x2.append (x[j])
        __y2.append (y[j])

    plot(matrix([__x1,__x2]),matrix([__y1,__y2]),c='k',linewidth=2)
    


def ReduceC(l,I,q):
    _l = []
    n = len(l)

    for e in range(0,n,2):
        _l.append(l[e])
        I[l[e],q+len(_l)-1] = 1.
        if e < n-2:
            I[l[e+1],q+len(_l)-1] = I[l[e+1],q+len(_l)] = .5
    if len(_l) > 1:
        I[l[n-1],q+len(_l)-1] = I[l[n-1],q] = .5
    
    return _l

# Coarsen a line, maintaining the endpoints
def ReduceL(l,I,q):
    _l = []
    n = len(l)

    if n % 2 == 0:
        for e in range(0,n-2,2):
            _l.append(l[e])
            I[l[e],q+len(_l)-1] = 1.
            if e < n-4:
                I[l[e+1],q+len(_l)-1] = I[l[e+1],q+len(_l)] = .5
        _l.append(l[n-1])
        I[l[n-1],q+len(_l)-1] = 1.
        I[l[n-3],q+len(_l)-2] = I[l[n-2],q+len(_l)-1] = 2./3.
        I[l[n-3],q+len(_l)-1] = I[l[n-2],q+len(_l)-2] = 1./3.        
    else:
        for e in range(0,n,2):
            _l.append(l[e])
            I[l[e],q+len(_l)-1] = 1.
            if e < n-2:
                I[l[e+1],q+len(_l)-1] = I[l[e+1],q+len(_l)] = .5
        
    return _l

# Coarsen a line, giving preference to the interior points
def ReduceI(l,I,q,e1,e2):
    _l = []
    n = len(l)

    if (n/2) % 2 == 1:
        for e in range(1,n/2,2):
            _l.append(l[e])
            I[l[e],q+len(_l)-1] = 1.
            #if e < n/2-2:
            I[l[e+1],q+len(_l)-1] = I[l[e+1],q+len(_l)] = .5
        I[l[0],q] = I[l[0],e1] = .5
    else:
        if n > 2:
            for e in range(0,n/2,2):
                _l.append(l[e])
                I[l[e],q+len(_l)-1] = 1.
                I[l[e+1],q+len(_l)-1] = I[l[e+1],q+len(_l)] = .5
        else:
            I[l[0],q] = I[l[0],e1] = .5
            

    for e in range(n/2,n,2):
        _l.append(l[e])
        I[l[e],q+len(_l)-1] = 1.
        if e < n-2:
            I[l[e+1],q+len(_l)-1] = I[l[e+1],q+len(_l)] = .5
    if (n-1-n/2) % 2 == 1 and n > 2:
        I[l[n-1],q+len(_l)-1] = I[l[n-1],e2] = .5

    return _l

def ReduceValve(s2, _x, _y, Links, Tethers, Parts):
    n = len(_x)
    I = zeros((n,n),float64)
    
    X, Y, newParts = [], [], []
    for p in range(len(Parts)-1):
        part = Parts[p]
        if p == 7 or p == 8:
            l = ReduceC(part,I,len(X))
        elif p == 5 or p == 6:# or p == 2 or p == 3 or p == 4:
            if p == 6:
                l = ReduceI(part,I,len(X),newParts[3][len(newParts[3])-1],newParts[2][len(newParts[2])-1])
            else:
                l = ReduceI(part,I,len(X),newParts[3][0],newParts[2][0])                
        else:
            l = ReduceL(part,I,len(X))

        _l = []
        for e in l:
            X.append(_x[e])
            Y.append(_y[e])
            _l.append(len(X)-1)
        newParts.append(_l)

##    clf()
##    imshow(I)
##    show()
##    raw_input("")

    newParts[6].reverse()
    newParts[3].reverse()
    newParts.append(newParts[2] + newParts[6] + newParts[3] + newParts[5])
    newParts[6].reverse()
    newParts[3].reverse()

    x2 = array(X)
    y2 = array(Y)

    newLinks = []
    for i in range(len(newParts[4])):
        v = newParts[4][i]
        if i > 0:
            _v = newParts[2][i-1]
            l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
            newLinks.append( (v, _v, s2, l) )

            _v = newParts[3][i-1]
            l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
            newLinks.append( (v, _v, s2, l) )
            
        if i < len(newParts[4])-1:
            _v = newParts[4][i+1]
            l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
            newLinks.append( (v, _v, s2, l) )

            _v = newParts[2][i+1]
            l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
            newLinks.append( (v, _v, s2, l) )

            _v = newParts[3][i+1]
            l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
            newLinks.append( (v, _v, s2, l) )

        _v = newParts[2][i]
        l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
        newLinks.append( (v, _v, s2, l) )

        _v = newParts[3][i]
        l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
        newLinks.append( (v, _v, s2, l) )

    _v = newParts[4][0]
    for v in newParts[5]:    
        l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
        newLinks.append( (v, _v, s2, l) )

    _v = newParts[4][len(newParts[4])-1]
    for v in newParts[6]:
        l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
        newLinks.append( (v, _v, s2, l) )


    # Add links about the surface of the valve
    for i in range(len(newParts[9])-1):
        v, _v = newParts[9][i], newParts[9][i+1]
        l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
        newLinks.append( (v, _v, s2, l) )
    v, _v = newParts[9][0], newParts[9][len(newParts[9])-1]
    l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
    newLinks.append( (v, _v, s2, l) )
        
        

    newTethers = []
    for e in newParts[0] + newParts[1] + newParts[7] + newParts[8]:
        newTethers.append( (e, (x2[e], y2[e]), s2) )

    return s2, I[:,:len(x2)], x2, y2, newLinks, newTethers, newParts





























def ReduceC2(l,I,q):
    _l = []
    n = len(l)

    for e in range(0,n,1):
        _l.append(l[e])
        I[l[e],q+len(_l)-1] = 1.
    
    return _l

# Coarsen a line, maintaining the endpoints
# All points with index less than N are kept, reduction above N
def ReduceL2(l,I,q,N):
    _l = []
    n = len(l)

    for e in range(0,N,1):
        _l.append(l[e])
        I[l[e],q+len(_l)-1] = 1.

    if (n-N) % 2 == 0:
        for e in range(N,n-2,2):
            _l.append(l[e])
            I[l[e],q+len(_l)-1] = 1.
            if e < n-4:
                I[l[e+1],q+len(_l)-1] = I[l[e+1],q+len(_l)] = .5
        _l.append(l[n-1])
        I[l[n-1],q+len(_l)-1] = 1.
        I[l[n-3],q+len(_l)-2] = I[l[n-2],q+len(_l)-1] = 2./3.
        I[l[n-3],q+len(_l)-1] = I[l[n-2],q+len(_l)-2] = 1./3.        
    else:
        for e in range(N,n,2):
            _l.append(l[e])
            I[l[e],q+len(_l)-1] = 1.
            if e < n-2:
                I[l[e+1],q+len(_l)-1] = I[l[e+1],q+len(_l)] = .5
        
    return _l


def ReduceValve2(s2, _x, _y, Links, Tethers, Parts, CutOff = .5):
    n = len(_x)
    I = zeros((n,n),float64)
    
    X, Y, newParts = [], [], []
    for p in range(len(Parts)-1):
        part = Parts[p]
        if p == 7 or p == 8:
            l = ReduceC2(part,I,len(X))
        elif p == 5 or p == 6:# or p == 2 or p == 3 or p == 4:
            if p == 6:
                l = ReduceI(part,I,len(X),newParts[3][len(newParts[3])-1],newParts[2][len(newParts[2])-1])
            else:
                l = ReduceI(part,I,len(X),newParts[3][0],newParts[2][0])                
        elif p == 2 or p == 3 or p == 4:
            # Find cutoff index
            N = 0
            for i in range(len(part)):
                if _y[part[i]] < CutOff:
                    N = i
            l = ReduceL2(part,I,len(X),N)
        else:
            l = ReduceL(part,I,len(X))

        _l = []
        for e in l:
            X.append(_x[e])
            Y.append(_y[e])
            _l.append(len(X)-1)
        newParts.append(_l)

##    clf()
##    imshow(I)
##    show()
##    raw_input("")

    newParts[6].reverse()
    newParts[3].reverse()
    newParts.append(newParts[2] + newParts[6] + newParts[3] + newParts[5])
    newParts[6].reverse()
    newParts[3].reverse()

    x2 = array(X)
    y2 = array(Y)

    newLinks = []
    for i in range(len(newParts[4])):
        v = newParts[4][i]
        if i > 0:
            _v = newParts[2][i-1]
            l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
            newLinks.append( (v, _v, s2, l) )

            _v = newParts[3][i-1]
            l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
            newLinks.append( (v, _v, s2, l) )
            
        if i < len(newParts[4])-1:
            _v = newParts[4][i+1]
            l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
            newLinks.append( (v, _v, s2, l) )

            _v = newParts[2][i+1]
            l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
            newLinks.append( (v, _v, s2, l) )

            _v = newParts[3][i+1]
            l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
            newLinks.append( (v, _v, s2, l) )

        _v = newParts[2][i]
        l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
        newLinks.append( (v, _v, s2, l) )

        _v = newParts[3][i]
        l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
        newLinks.append( (v, _v, s2, l) )

    _v = newParts[4][0]
    for v in newParts[5]:    
        l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
        newLinks.append( (v, _v, s2, l) )

    _v = newParts[4][len(newParts[4])-1]
    for v in newParts[6]:
        l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
        newLinks.append( (v, _v, s2, l) )


    # Add links about the surface of the valve
    for i in range(len(newParts[9])-1):
        v, _v = newParts[9][i], newParts[9][i+1]
        l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
        newLinks.append( (v, _v, s2, l) )
    v, _v = newParts[9][0], newParts[9][len(newParts[9])-1]
    l = ((x2[v] - x2[_v])**2 + (y2[v] - y2[_v])**2)**.5
    newLinks.append( (v, _v, s2, l) )
        
        

    newTethers = []
    for e in newParts[0] + newParts[1] + newParts[7] + newParts[8]:
        newTethers.append( (e, (x2[e], y2[e]), s2) )

    return s2, I[:,:len(x2)], x2, y2, newLinks, newTethers, newParts



def ReduceValve3(_x, _y, Parts, Off=0):
    n = len(_x)
    I = zeros((n,n),float64)
    
    X, Y, newParts = [], [], []
    for p in range(len(Parts)):
        part = Parts[p]
        if p == 7 or p == 8 or p == 9:
            if Off == 1:
                l = ReduceC(part[1:]+part[:1],I,len(X))
            else:
                l = ReduceC(part,I,len(X))
        elif p ==0 or p == 1:
            l = ReduceL(part,I,len(X))
        elif p == 4:
            l = [part[0], part[len(part)-1]]
            I[part[0],len(X)] = 1.
            I[part[len(part)-1],len(X)+1] = 1.
        else:
            l = []

        _l = []
        for e in l:
            X.append(_x[e])
            Y.append(_y[e])
            _l.append(len(X)-1)
        newParts.append(_l)

    x2 = array(X)
    y2 = array(Y)

    return I[:,:len(x2)], x2, y2, newParts




####s = 100000
####x, y, Links, Tethers, Parts = ConstructValve (s, 0., 2., 0., 1., .005,False,1.33,1.)#,1.5,1.15,False)
#####clf(); scatter(x,y); raw_input("")
############
############
####clf()
####scatter(x,y,s=4,c='k',marker='o')
####axis([0, 2, 0, 1])
####raise
####I,x2,y2,Parts2 = ReduceValve3(x,y,Parts)
#####clf(); PlotValve(y,x,Links)
####clf(); scatter(y,x,marker='o',s=20)
####scatter(y2,x2,marker='o',s=45,c='k')
####axis([.22, .79, .268, .392])
####raise


########scatter(x2,y2,c='r'); raw_input("")
########
########I,x2,y2,Parts2 = ReduceValve3(x,y,Parts)
########scatter(x2,y2,c='g'); raw_input("")

####I2,x3,y3,Parts3 = ReduceValve3(x2,y2,Parts2)
####scatter(x3,y3,c='g'); raw_input("")
####I3,x4,y4,Parts4 = ReduceValve3(x3,y3,Parts3)
####scatter(x4,y4,c='k'); raw_input("")
####
####x,y = dot(I,dot(I2,dot(I3,x4))), dot(I,dot(I2,dot(I3,y4)))
####scatter(x,y,c='y'); raw_input("")

##clf()
##PlotValve(_x,_y,Links)
##clf()
##for i in range(len(Parts)):
##    p = Parts[i]
##    if i == 0 or i == 1 or i == 7 or i == 8:
##        scatter(x[p],y[p],s=21,c='k',marker='x')
##    else:
##        scatter(x[p],y[p],c='k',s=14)
##PlotValve(x,y,Links)
##axis([0, 2, 0, 1])
#####PlotValve(_x,_y+ones(len(_y),float64),[])
#####PlotValve(_x+2.*ones(len(_x),float64),_y,[])
####show()
####
####

##s2, I, _x2, _y2, Links2, Tethers2, Parts2 = ReduceValve2(s, _x, _y, Links, Tethers, Parts)
##s3, I2, _x3, _y3, Links3, Tethers3, Parts3 = ReduceValve2(s2, _x2, _y2, Links2, Tethers2, Parts2)

##clf()
##scatter(_x,_y,marker='x',c='k',s=11)
##scatter(_x2,_y2,c='k',s=15)
##axis([.2,.55,.23,.8])

##clf()
##scatter(_x,_y,s=7,c='y')
##raw_input("")
##scatter(_x2,_y2,s=10,c='b')
##raw_input("")
##scatter(_x3,_y3,s=13,c='r')



##_I = 1.*transpose(I)
##n,m = _I.shape
##for i in range (n):
##    for j in range (m):
##        if _I[i,j] != 1.: _I[i,j] = 0.
##_x2 = dot(_I,_x)
##_y2 = dot(_I,_y)
##
##x = dot(I,_x2)
##y = dot(I,_y2)
##clf()
##scatter(_x2,_y2,s=16,c='b')
##raw_input("")
##scatter(x,y,s=9,c='y')
##raw_input("")
####
####
####
####s3, I, _x3, _y3, Links3, Tethers3, Parts3 = ReduceValve(s2, _x2, _y2, Links2, Tethers2, Parts2)
####s4, I, _x4, _y4, Links4, Tethers4, Parts4 = ReduceValve(s3, _x3, _y3, Links3, Tethers3, Parts3)
####
####
####clf()
####scatter(_x,_y,s=7,c='y')
####show(); raw_input("")
####scatter(_x2,_y2,s=10,c='b')
####show(); raw_input("")
####scatter(_x3,_y3,s=13,c='r')
####show(); raw_input("")
####scatter(_x4,_y4,s=16,c='g')
####show(); raw_input("")
####PlotValve(_x4,_y4,Links4)
####
####
####for v in Parts2[9]:
####    scatter([_x2[v]],[_y2[v]],s=30,c='k')
####    show()
####    raw_input("")
