from Plateau_Problem.Triangulation_Meshing.Area_minimizing import *
from typing import *



def projection(Tr,P):
    "projete le point x sur le triangle tr"
    # On projette les deux points sur le plan du triangle et on s'assure que c'est dedans
    A = Tr[0]
    B = Tr[1]
    C = Tr[2]
    #Calcul du vecteur normal
    N = np.cross(B-A,B-C)
    n = N/np.linalg.norm(N)
    d = np.dot(n,P-A)
    return P - d*n

def area_of_triangle(Tr):
    "calcule l'aire d'un triangle array"
    p1, p2, p3 = Tr[0], Tr[1], Tr[2]
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

def intersection(Tr,P,Q):
    "determine l'intersection entre PQ et le plan du triangle Tr "
    A = projection(Tr,P)
    B = projection(Tr,Q)
    # on che rche l'intersectionn entre BA et PQ
    a,a_p = tuple(list(B-A)[:2])
    b,b_p = tuple(list(P-Q)[:2])
    c,c_p = tuple(list(P-A)[:2])
    t = (a*c_p - a_p*c)/(a*b_p - a_p*b)
    lbda = (c*b_p - c_p*b)/(a*b_p - a_p*b)
    assert np.isclose(P +t*(Q-P), A+lbda*(B-A)).all() , "error in the code"
    return P +t*(Q-P)


def is_inside(Tr,P): 
    "determine si un point P est à l'intérieur d'un triangle Tr"
    A, B, C = Tr[0], Tr[1], Tr[2]
    Tr1 = [A,B,P]
    Tr2 = [A,C,P]
    Tr3 = [B,C,P]
    Somme = area_of_triangle(Tr1) + area_of_triangle(Tr2) + area_of_triangle(Tr3)
    Total = area_of_triangle(Tr)
    # assert(Total <= Somme), "Error dans le test"
    return np.isclose(Somme ,Total)

def traverse(Tr,P,Q):
    "determine si PQ traverse Tr"
    # les points P et Q  doivent etre dans deux sens opposés de de tr
    P_proj = projection(Tr,P)
    Q_proj = projection(Tr,Q)
    n_P = (P - P_proj)
    n_Q = (Q - Q_proj)
    if np.dot(n_P,n_Q) > 0 :
        return False
    Z = intersection(Tr,P,Q)
    return is_inside(Tr,Z) 


class Block_edges():
    def __init__(self, triangles : List[Tuple[int, int, int]], mapping : List[np.ndarray] ):
        """
        The meshing contains indexes of the triangles (i,j,k)
        andf the mapping givers the coordiantes of each point mapping[i] = array([x,y,z])
        """
        self.triangles = triangles 
        self.mapping = mapping

    



