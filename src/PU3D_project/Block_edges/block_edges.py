from Plateau_Problem.Triangulation_Meshing.Area_minimizing import *
from typing import *

def cross(A,B):
    assert len(A) == len(B) == 3
    a1,a2,a3 = A[0],A[1],A[2]
    b1,b2,b3 = B[0],B[1],B[2]
    return np.array([a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]) 

def norm(A):
    a1,a2,a3 = A[0],A[1],A[2]
    return np.sqrt(a1**2 + a2**2 + a3**2)


def projection(Tr,P):
    "projete le point x sur le triangle tr"
    # On projette les deux points sur le plan du triangle et on s'assure que c'est dedans
    A = Tr[0]
    B = Tr[1]
    C = Tr[2]
    #Calcul du vecteur normal
    N = cross(B-A,B-C)
    n = N/norm(N)
    d = np.dot(n,P-A)
    return P - d*n

def norm(A):
    a1,a2,a3 = A[0],A[1],A[2]
    return np.sqrt(a1**2 + a2**2 + a3**2)

def area_of_triangle(Tr):
    "calcule l'aire d'un triangle array"
    p1, p2, p3 = Tr[0], Tr[1], Tr[2]
    return 0.5 * norm(cross(p2 - p1, p3 - p1))

def intersection(Tr,P,Q):
    "determine l'intersection entre PQ et le plan du triangle Tr "
    A = projection(Tr,P)
    B = projection(Tr,Q)
    if np.isclose(A,B).all():
        return A
    if (A == P).all() or (A == Q).all():
        return A
    if (B == P).all() or (B == Q).all():
        return B
    
    # on cherche l'intersectionn entre BA et PQ
    else:
        ind = 2
        if 0 in P-Q :
            ind = np.where(P-Q == 0)[0][0]
        a,a_p = tuple(list(B-A)[:ind]+list(B-A)[ind+1:])
        b,b_p = tuple(list(P-Q)[:ind]+list(P-Q)[ind+1:])
        c,c_p = tuple(list(P-A)[:ind]+list(P-A)[ind+1:])
        if (a*b_p - a_p*b) == 0:
            ind = np.where(P-Q == 0)[0][1]
        a,a_p = tuple(list(B-A)[:ind]+list(B-A)[ind+1:])
        b,b_p = tuple(list(P-Q)[:ind]+list(P-Q)[ind+1:])
        c,c_p = tuple(list(P-A)[:ind]+list(P-A)[ind+1:])
        t = (a*c_p - a_p*c)/(a*b_p - a_p*b)
        lbda = (c*b_p - c_p*b)/(a*b_p - a_p*b)
        assert np.isclose(P +t*(Q-P), A+lbda*(B-A),atol =1e-10).all() ,("Tr",Tr,P +t*(Q-P), A+lbda*(B-A),"P",P,"Q",Q,"A",A,"B",B)
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
    if np.isclose(P_proj,P).all() and np.isclose(Q_proj,Q).all():
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
        self.blocked_edges = [] # contains the blocked edges as pairs of (P,Q) P and Q being two points.

    def detect_edges(self,tr):   
        
        """Gives the edges in the 3D grid 
        that go through a triangle tr"""

        x_coords = [self.mapping[point][0] for point in tr]
        y_coords = [self.mapping[point][1] for point in tr]
        z_coords = [self.mapping[point][2] for point in tr]
        
        Tr = [np.array(self.mapping[pt]) for pt in tr]
        if  (cross(Tr[0] - Tr[1], Tr[0] - Tr[2] )==0).all():
            return None
        # Calculate min and max for each dimension with integer bounds
        x_min = math.floor(min(x_coords))
        y_min = math.floor(min(y_coords))
        z_min = math.floor(min(z_coords))

        x_max = math.ceil(max(x_coords))
        y_max = math.ceil(max(y_coords))
        z_max = math.ceil(max(z_coords))   

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for z in range(z_min, z_max + 1):
                    # Add edges parallel to the x-axis
                    if x <= x_max:
                        P,Q = np.array((x, y, z)), np.array((x + 1, y, z))
                        if traverse(Tr,P,Q):
                            self.blocked_edges.append((P,Q))
                    # Add edges parallel to the y-axis
                    if y <= y_max:
                        P,Q = np.array((x, y, z)), np.array((x, y + 1, z))
                        
                        if traverse(Tr,P,Q):
                            self.blocked_edges.append((P,Q))
                    # Add edges parallel to the z-axis
                    if z <= z_max:
                        P,Q = np.array((x, y, z)), np.array((x, y, z + 1))
                        if traverse(Tr,P,Q):
                            self.blocked_edges.append((P,Q))
                    # if x >= x_min:
                    #     P,Q = np.array((x, y, z)), np.array((x - 1, y, z))
                    #     if traverse(Tr,P,Q):
                    #         self.blocked_edges.append((P,Q))
                    # if y >= y_min:
                    #     P,Q = np.array((x, y, z)), np.array((x, y - 1, z))
                    #     if traverse(Tr,P,Q):
                    #         self.blocked_edges.append((P,Q))
                    # if z >= z_min:
                    #     P,Q = np.array((x, y, z)), np.array((x, y, z - 1))
                    #     if traverse(Tr,P,Q):
                    #         self.blocked_edges.append((P,Q))

    
    def block_all_the_edges(self):
        for tr in (self.triangles):
            self.detect_edges(tr)



    



    



