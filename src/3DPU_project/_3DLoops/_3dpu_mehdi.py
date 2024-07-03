import numpy as np
import random
from copy import *

dim = int(3)

M = np.array([
    [0,0,0,0],
    [0,1,0,0],
    [1,0,0,0],
    [1,0,1,0],
    [2,0,0,0],
    [2,0,0,1]
])

M_x = np.array([[1,1,1,0,0],
                [1,1,-1,0,0],
                [2,-1,0,0,0],
                [2,1,0,1,0],
                [2,1,-1,0,0],
                [2,-1,-1,1,0],
                [3,-1,0,0,0],
                [3,1,0,0,1],
                [3,1,-1,0,0],
                [3,-1,-1,0,1]])

M_y = np.array([[1,-1,0,0,0],
                [1,1,1,0,0],
                [1,1,0,-1,0],
                [1,-1,1,-1,0],
                [2,1,0,1,0],
                [2,1,0,-1,0],
                [3,-1,0,0,0],
                [3,1,0,0,1],
                [3,1,0,-1,0],
                [3,-1,0,-1,1]])
        
M_z = np.array([[1,-1,0,0,0],
                [1,1,1,0,0],
                [1,1,0,0,-1],
                [1,-1,1,0,-1],
                [2,-1,0,0,0],
                [2,1,0,1,0],
                [2,1,0,0,-1],
                [2,-1,0,1,-1],
                [3,1,0,0,1],
                [3,1,0,0,-1]])


def wrap(phi):
    return np.round(phi / (2 * np.pi)).astype(int)


def grad(psi, a: int):
    return np.diff(psi, axis=a)


def wrap_grad(psi, a: int):
    return wrap(grad(psi, a))


def residuals(psi, a: int):
    assert(a >= 0 and a < dim)
    ax, ay = (1 + np.arange(dim-1) + a) % dim
    gx = wrap_grad(psi, a=ax)
    gy = wrap_grad(psi, a=ay)
    return grad(gy, a=ax) - grad(gx, a=ay)


def verify(indice,l):
    ind = np.array(indice)
    shpe = np.array(l).shape
    return (ind >= 0).all() and (ind < shpe).all()


def au_bord_pos(s,S):
    shpe = np.array(S).shape
    au_bord = False
    if s[0] == 0:
        return (s[1] == 0) or (s[1] == shpe[1]-1)
    if s[0] == 1:
        return (s[2] == 0) or (s[2] == shpe[2]-1)
    if s[0] == 2:
        return (s[3] == 0) or (s[3] == shpe[3]-1)


def are_neighbours(s1,s2):
    [ax,i,j,k] = s2
    booleen = False
    if ax == 0 :
        for mx in M_x:
            ind = (mx[0]-1,mx[2]+i,mx[3]+j,mx[4]+k)
            booleen = (s1 == ind) or booleen
    
    elif ax == 1 :
        for my in M_y:
            ind = (my[0]-1,my[2]+i,my[3]+j,my[4]+k)  
            booleen = (s1 == ind) or booleen

    elif ax == 2 :
        for mz in M_z: 
            ind = (mz[0]-1,mz[2]+i,mz[3]+j,mz[4]+k)
            booleen = (s1 == ind) or booleen
    
    return booleen


def same_cube(s1,s2):
    v1 = np.array(s1)
    v2 = np.array(s2)
    D = copy(M)
    D[:,1:] = D[:,1:] + v1[1:]
    for d in D:
        if (v2 == d).all():
            return True
    return False


def get_neighbour(deb,St) :
    voisins = []
    [ax,i,j,k] = deb

    if ax == 0:
        for mx in M_x:
            ind = mx[0]-1,mx[2]+i,mx[3]+j,mx[4]+k
            if verify(ind,St) and St[ind] != 0 and (St[ind] == mx[1]*St[ax,i,j,k]) :
                voisins.append(ind)
                
    if ax == 1 :
        for my in M_y :
            ind  = my[0]-1,my[2]+i,my[3]+j,my[4]+k
            if verify(ind,St) and(St[ind] != 0) and St[ind] == my[1]*St[ax,i,j,k]  :
                voisins.append(ind)
                
    if ax == 2 :
        for mz in M_z :
            ind = mz[0]-1,mz[2]+i,mz[3]+j,mz[4]+k
            if verify(ind,St) and (St[ind] != 0) and St[ind] == mz[1]*St[ax,i,j,k] :
                voisins.append(ind)
                
    return voisins


def all_residuals(psi):
    rx = residuals(psi, 0)
    ry = residuals(psi, 1)
    rz = residuals(psi, 2)

    St = np.zeros((dim,psi.shape[0],psi.shape[1],psi.shape[2]))
    St[0,:rx.shape[0],:rx.shape[1],:rx.shape[2]] = rx
    St[1,:ry.shape[0],:ry.shape[1],:ry.shape[2]] = ry
    St[2,:rz.shape[0],:rz.shape[1],:rz.shape[2]] = rz
    return St


def not_knot(ch):
    booleen = False
    for i in range(len(ch)-2):
        booleen = booleen or (are_neighbours(ch[i],ch[i+2]))
    return not booleen


def residual_loops(chemins,St): 
    original_St = copy(St)
    while (St != 0).any():
        # Initilize the start position
        chemin = []
        ax,x,y,z = np.where(St != 0)
        ind = np.random.randint(len(ax))
        deb = ax[ind],x[ind],y[ind],z[ind]
        suivant = deb
        init = deb
        print("new il reste{}".format(len(ax)))
        chemin.append(deb)
        #getting first loop


        while get_neighbour(deb,St) != [] :
            suivants = get_neighbour(deb,St)
            random_index = random.randint(0,len(suivants)-1)
            suivant = suivants[random_index]
            St[deb] = 0
            chemin.append(suivant)
            deb = suivant

        #seeing if it a closed loop
        if init in get_neighbour(deb,original_St):
            if  len(chemin) > 3 and not_knot(chemin):
                chemin.append(init)  
                chemins.append((1,chemin))
                St[deb] = 0
                St[init] = 0
            #the only case the loop is not 3 length is if 
            elif au_bord_pos(chemin[0],St) and au_bord_pos(chemin[-1],St):
                chemins.append((0,chemin)) 
                St[deb] = 0
                St[init] = 0

            else :
                for k in chemin :
                    St[k] = original_St[k]
        # if not a closed loop 
        else:
            # we continue in the other direction meaning from the position init 
            deb = init
            suivant = deb

            while get_neighbour(deb,St) != []:
                suivants = get_neighbour(deb,St)
                St[deb] = 0
                random_index = random.randint(0,len(suivants)-1)
                suivant = suivants[random_index]
                cpt += 1
                chemin = [suivant] + chemin
                deb = copy(suivant)


            if not_knot(chemin) and (len(chemin) > 1) and au_bord_pos(chemin[0],St) and au_bord_pos(chemin[-1],St) :
                chemins.append((0,chemin))
                St[deb] = 0
                St[chemin[-1]] = 0

            else :
                for k in chemin :
                    St[k] = original_St[k]
                
                    
    return St



if __name__ == '__main__':
    pass