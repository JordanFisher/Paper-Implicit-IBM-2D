from pylab import *

# Determines if a point (x, y) is in a polygon
def IsIn (x, y, polyx, polyy):
    project = (x - polyx[3]) * (polyy[0] - polyy[3]) - (y - polyy[3]) * (polyx[0] - polyx[3])
    if project <= 0:
        project = (x - polyx[1]) * (polyy[2] - polyy[1]) - (y - polyy[1]) * (polyx[2] - polyx[1])
        if project <= 0:
            project = (x - polyx[2]) * (polyy[3] - polyy[2]) - (y - polyy[2]) * (polyx[3] - polyx[2])
            if project <= 0:
                project = (x - polyx[0]) * (polyy[1] - polyy[0]) - (y - polyy[0]) * (polyx[1] - polyx[0])
                if project <= 0:
                    return True

    return False

# Find the best bounding box for a polygon
def BoundingBox (h, polyx, polyy):
    minj = int(min(polyx) / h) + 1
    maxj = int(max(polyx) / h) + 1
    mink = int(min(polyy) / h) + 1
    maxk = int(max(polyy) / h) + 1

    return minj, maxj, mink, maxk


def DoubleSidedNormalSquare (Gamma, x1, x2, y1, y2, tx, ty, polyx, polyy):
    polyx[0] = x1 - Gamma * ty
    polyx[3] = x2 - Gamma * ty
    polyx[2] = x2 + Gamma * ty
    polyx[1] = x1 + Gamma * ty
    polyy[0] = y1 + Gamma * tx
    polyy[3] = y2 + Gamma * tx
    polyy[2] = y2 - Gamma * tx
    polyy[1] = y1 - Gamma * tx

def Wedge (Gamma, x, y, tx1, ty1, tx2, ty2, polyx, polyy):
    # Check to see which side to put the wedge on
    if tx1 * ty2 - ty1 * tx2 > 0:
        polyx[0] = x - Gamma * ty1
        polyx[2] = x - Gamma * ty2
        polyy[0] = y + Gamma * tx1
        polyy[2] = y + Gamma * tx2

        if abs(tx1 + tx2) > .1:
            c = -(polyx[2] - polyx[0]) / (tx1 + tx2)
        else:
            c = -(polyy[2] - polyy[0]) / (ty1 + ty2)

        polyx[1] = polyx[0] - tx1 * c
        polyx[3] = x
        polyy[1] = polyy[0] - ty1 * c
        polyy[3] = y

        return -1
    else:
        polyx[0] = x + Gamma * ty1
        polyx[2] = x + Gamma * ty2
        polyy[0] = y - Gamma * tx1
        polyy[2] = y - Gamma * tx2

        if abs(tx1 + tx2) > .1:
            c = (polyx[2] - polyx[0]) / (tx1 + tx2)
        else:
            c = (polyy[2] - polyy[0]) / (ty1 + ty2)

        polyx[3] = polyx[0] + tx1 * c
        polyx[1] = x
        polyy[3] = polyy[0] + ty1 * c
        polyy[1] = y

        return 1

def OneSidedNormalSquare (side, Gamma, x1, x2, y1, y2, tx, ty, polyx, polyy):
    if side == 1:
        polyx[0] = x1 - Gamma * ty
        polyx[3] = x2 - Gamma * ty
        polyx[2] = x2
        polyx[1] = x1
        polyy[0] = y1 + Gamma * tx
        polyy[3] = y2 + Gamma * tx
        polyy[2] = y2
        polyy[1] = y1
    else:
        polyx[0] = x1 + Gamma * ty
        polyx[1] = x2 + Gamma * ty
        polyx[2] = x2
        polyx[3] = x1
        polyy[0] = y1 - Gamma * tx
        polyy[1] = y2 - Gamma * tx
        polyy[2] = y2
        polyy[3] = y1

def OneSidedBisectorSquare (side, Gamma, x1, x2, y1, y2, bx1, by1, bx2, by2, polyx, polyy):
    if side == 1:
        polyx[0] = x1 + bx1
        polyx[1] = x2 + bx2
        polyx[2] = x2
        polyx[3] = x1
        polyy[0] = y1 + by1
        polyy[1] = y2 + by2
        polyy[2] = y2
        polyy[3] = y1
    else:
        polyx[0] = x1 - bx1
        polyx[3] = x2 - bx2
        polyx[2] = x2
        polyx[1] = x1
        polyy[0] = y1 - by1
        polyy[3] = y2 - by2
        polyy[2] = y2
        polyy[1] = y1
        

def TwoSidedBisectorSquare (Gamma, x1, x2, y1, y2, bx1, by1, bx2, by2, polyx, polyy):
    polyx[0] = x1 + bx1
    polyx[1] = x2 + bx2
    polyx[2] = x2 - bx2
    polyx[3] = x1 - bx1
    polyy[0] = y1 + by1
    polyy[1] = y2 + by2
    polyy[2] = y2 - by2
    polyy[3] = y1 - by1


# ret = 0 returns the extended velocity
# ret = 1 returns the distance metric
def OldFPT (U, V, N, M, Nb, h, Gamma, X, Y, ret = 0):
    d = zeros((N,M), float64)
    u = zeros((N,M), float64)
    v = zeros((N,M), float64)
    tx = zeros(Nb, float64)
    ty = zeros(Nb, float64)
    length = zeros(Nb, float64)

    # Initialize polygon array
    PolygonsX = []
    PolygonsY = []
    for i in range(2*Nb):
        PolygonsX.append ([0, 0, 0, 0])
        PolygonsY.append ([0, 0, 0, 0])    

    # Compute tangent vectors directions
    for j in range(Nb-1):
        tx[j] = X[j+1] - X[j]
        ty[j] = Y[j+1] - Y[j]
        length[j] = (tx[j]**2 + ty[j]**2)**.5
    tx[Nb-1] = X[0] - X[Nb-1]
    ty[Nb-1] = Y[0] - Y[Nb-1]
    length[Nb-1] = (tx[Nb-1]**2 + ty[Nb-1]**2)**.5

    # Normalize the tangent vectors
    tx /= length
    ty /= length
    

    # Compute normal boxes
    for j in range(0,Nb-1):
        DoubleSidedNormalSquare (Gamma, X[j], X[j+1], Y[j], Y[j+1], tx[j], ty[j], PolygonsX[j], PolygonsY[j])
    DoubleSidedNormalSquare (Gamma, X[Nb-1], X[0], Y[Nb-1], Y[0], tx[Nb-1], ty[Nb-1], PolygonsX[Nb-1], PolygonsY[Nb-1])

    mult = []
    mult.append (Wedge (Gamma, X[0], Y[0], tx[0], ty[0], tx[Nb-1], ty[Nb-1], PolygonsX[Nb], PolygonsY[Nb]))
    for j in range(1, Nb):
        mult.append (Wedge (Gamma, X[j], Y[j], tx[j], ty[j], tx[j-1], ty[j-1], PolygonsX[j + Nb], PolygonsY[j + Nb]))

    for q in range(2):
        # Find points in polys
        for i in range(2 * Nb):
            print q, i, h, PolygonsX[i], PolygonsY[i]
            minj, maxj, mink, maxk = BoundingBox (h, PolygonsX[i], PolygonsY[i])
            minj = max(0,minj)
            mink = max(0,mink)
            maxj = min(N,maxj)
            maxk = min(M,maxk)
            for j in range(minj, maxj):
                for k in range(mink, maxk):
                    # Check to see if points are in polygon i
                    x, y = j * h, k * h

                    if IsIn (x, y, PolygonsX[i], PolygonsY[i]):                    
                        if q == 0:
                            d[j,k] = Gamma
                        else:
                            if i < Nb:
                                # Find distance by projection
                                dist = (x - X[i]) * ty[i] - (y - Y[i]) * tx[i]

                                if abs(dist) < abs(d[j,k]):
                                    d[j,k] = dist

                                    dist = (x - X[i]) * tx[i] + (y - Y[i]) * ty[i]
                                    if i == Nb-1:
                                        u[j,k] = U[i] + (U[0] - U[i]) * dist / length[i]
                                        v[j,k] = V[i] + (V[0] - V[i]) * dist / length[i]
                                    else:
                                        u[j,k] = U[i] + (U[i+1] - U[i]) * dist / length[i]
                                        v[j,k] = V[i] + (V[i+1] - V[i]) * dist / length[i]
                            else:
                                # Find distance to vertex
                                dist = mult[i - Nb] * ((x - X[i - Nb])**2 + (y - Y[i - Nb])**2)**.5

                                if abs(dist) < abs(d[j,k]):
                                    d[j,k] = dist
                                
                                    u[j,k] = U[i - Nb]
                                    v[j,k] = V[i - Nb]

    if ret == 0:
        return u, v
    else:
        return d

def NewFPT (u, v, U, V, N, Nb, Gamma, X, Y, tx, ty, length, PolygonsX, PolygonsY):
    bx = zeros(Nb, float64)
    by = zeros(Nb, float64)

    d = zeros((N,N), float64)
#    u = zeros((N,N), float64)
#    v = zeros((N,N), float64)

    # Compute the angle bisectors
    for j in range(1,Nb):
        bx[j] = ty[j] + ty[j-1]
        by[j] = -(tx[j] + tx[j-1])
    bx[0] = ty[0] + ty[Nb-1]
    by[0] = -(tx[0] + tx[Nb-1])

    # Extend bisectors to required length
    for j in range(Nb):
        mult = Gamma / (bx[j] * ty[j] - by[j] * tx[j])
        bx[j] *= mult
        by[j] *= mult

        projs[j] = bx[j] * tx[j] + by[j] * ty[j]

    # Check for crossovers and make necessary additional polygons
    index = 0
    for j in range(Nb-1):
        width = projs[j] + projs[j+1]
        if width < -length[j]:
            OneSidedNormalSquare (1, Gamma, X[j], X[j+1], Y[j], Y[j+1], tx[j], ty[j], PolygonsX[Nb + index], PolygonsY[Nb + index])
            index += 1

            OneSidedBisectorSquare (1, Gamma, X[j], X[j+1], Y[j], Y[j+1], bx[j], by[j], bx[j+1], by[j+1], PolygonsX[j], PolygonsY[j])
        elif width > length[j]:
            OneSidedNormalSquare (-1, Gamma, X[j], X[j+1], Y[j], Y[j+1], tx[j], ty[j], PolygonsX[Nb + index], PolygonsY[Nb + index])
            index += 1

            OneSidedBisectorSquare (-1, Gamma, X[j], X[j+1], Y[j], Y[j+1], bx[j], by[j], bx[j+1], by[j+1], PolygonsX[j], PolygonsY[j])
        else:
            TwoSidedBisectorSquare (Gamma, X[j], X[j+1], Y[j], Y[j+1], bx[j], by[j], bx[j+1], by[j+1], PolygonsX[j], PolygonsY[j])
    
    width = projs[Nb-1] + projs[0]
    if width < -length[Nb-1]:
        OneSidedNormalSquare (1, Gamma, X[Nb-1], X[0], Y[Nb-1], Y[0], tx[Nb-1], ty[Nb-1], PolygonsX[Nb + index], PolygonsY[Nb + index])
        index += 1

        OneSidedBisectorSquare (1, Gamma, X[Nb-1], X[0], Y[Nb-1], Y[0], bx[Nb-1], by[Nb-1], bx[0], by[0], PolygonsX[Nb-1], PolygonsY[Nb-1])
    elif width > length[Nb-1]:
        OneSidedNormalSquare (-1, Gamma, X[Nb-1], X[0], Y[Nb-1], Y[0], tx[Nb-1], ty[Nb-1], PolygonsX[Nb + index], PolygonsY[Nb + index])
        index += 1

        OneSidedBisectorSquare (-1, Gamma, X[Nb-1], X[0], Y[Nb-1], Y[0], bx[Nb-1], by[Nb-1], bx[0], by[0], PolygonsX[Nb-1], PolygonsY[Nb-1])
    else:
        TwoSidedBisectorSquare (Gamma, X[Nb-1], X[0], Y[Nb-1], Y[0], bx[Nb-1], by[Nb-1], bx[0], by[0], PolygonsX[Nb-1], PolygonsY[Nb-1])


    
    # Flag points in pathological polys
    for i in range(index):
        minj, maxj, mink, maxk = BoundingBox (PolygonsX[i + Nb], PolygonsY[i + Nb])    
        for j in range(minj, maxj):
            for k in range(mink, maxk):
                # Check to see if points are in polygon i
                x, y = j * h, k * h
                if IsIn (x, y, PolygonsX[i + Nb], PolygonsY[i + Nb]):
                    d[j,k] = Gamma

    # Check points in normal polys    
    for i in range(Nb):
        minj, maxj, mink, maxk = BoundingBox (PolygonsX[i], PolygonsY[i])    
        for j in range(minj, maxj):
            for k in range(mink, maxk):
                # Check to see if points are in polygon i
                x, y = j * h, k * h
                
                if IsIn (x, y, PolygonsX[i], PolygonsY[i]):
                    dist = (x - X[i]) * tx[i] + (y - Y[i]) * ty[i]

                    if dist < 0:
                        u[j,k] = U[i]
                        v[j,k] = V[i]
                    elif dist > length[i]:
                        if i == Nb-1:
                            u[j,k] = U[0]
                            v[j,k] = V[0]
                        else:
                            u[j,k] = U[i+1]
                            v[j,k] = V[i+1]
                    else:
                        if i == Nb-1:
                            u[j,k] = U[i] + (U[0] - U[i]) * dist / length[i]
                            v[j,k] = V[i] + (V[0] - V[i]) * dist / length[i]
                        else:
                            u[j,k] = U[i] + (U[i+1] - U[i]) * dist / length[i]
                            v[j,k] = V[i] + (V[i+1] - V[i]) * dist / length[i]
                    
                    print u[j,k]
                    d[j,k] = dist

    # Recheck points in pathological polys
    for i in range(index):
        minj, maxj, mink, maxk = BoundingBox (PolygonsX[i + Nb], PolygonsY[i + Nb])    
        for j in range(minj, maxj):
            for k in range(mink, maxk):
                # Check to see if points are in polygon i
                x, y = j * h, k * h
                if IsIn (x, y, PolygonsX[i + Nb], PolygonsY[i + Nb]):
                    # Find distance by projection
                    dist = (x - X[i]) * ty[i] - (y - Y[i]) * tx[i]

                    if abs(dist) < abs(d[j,k]):
                        d[j,k] = dist

                        dist = (x - X[i]) * tx[i] + (y - Y[i]) * ty[i]
                        
                        if i == Nb-1:
                            u[j,k] = U[i] + (U[0] - U[i]) * dist / length[i]
                            v[j,k] = V[i] + (V[0] - V[i]) * dist / length[i]
                        else:
                            u[j,k] = U[i] + (U[i+1] - U[i]) * dist / length[i]
                            v[j,k] = V[i] + (V[i+1] - V[i]) * dist / length[i]















