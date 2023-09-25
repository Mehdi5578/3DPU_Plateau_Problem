from PointList import *
from Triangle import *


class TriangularMesh:
    def __init__(self, boundary : PointList, desired_triangle_count):
        self.boundary = boundary 
        self.m = desired_triangle_count
        self.mesh = []
        self.n = len(boundary.points)
        self.v_indexes = [] #the vertexes indices
        self.mapping = [] #contains the mapping from I(v_indexes to R^3)
        self.triangles  = []  #each triangle is 
        

    def compute_central_point(self):
        return self.boundary.average_point()
        
         
    def create_initial_subdivisions(self):
        #  Create initial subdivisions using the central point
        C = self.compute_central_point()
        s = int(self.m/(2*self.n) - 1/2)
        triangle = self.boundary.points + [self.boundary.points[0]]
        P = [triangle]
        for j in range(1,s+1):
            P_j = triangle + (j / (s+1))*(C - triangle)
            P.append(P_j)
        
        return np.array(P)
        

    def create_quadrilaterals(self):
        #split the outside quadrilaterals
        P = (self.create_initial_subdivisions())
        s = int(self.m/(2*self.n) - 1/2)
        for j in tqdm(range(s)) : 
            for i in range(self.n) :
                self.mesh.append([P[j,i],P[j+1,i+1],P[j,i+1]])
                self.mesh.append([P[j,i],P[j+1,i+1],P[j+1,i]])
        
        


    def split_quadrilateral(self):
        C = self.compute_central_point()
        P = self.create_initial_subdivisions()
        for i in tqdm(range(self.n)) :
            self.mesh.append([C,P[-1,i],P[-1,i+1]])

    def further_subdivide(self):
        while len(self.mesh) < self.m :
            k = np.random.randint(0,len(self.mesh))
            tri = np.array(self.mesh.pop(k))
            C = np.mean(tri,axis = 0)
            tri1 = [tri[0],tri[1],C]
            tri2 = [tri[0],tri[2],C]
            tri3 = [tri[1],tri[2],C]
            self.mesh += [tri1,tri2,tri3]
    



    def generate_mesh_initial(self):
        self.create_quadrilaterals()
        self.split_quadrilaterals()
        self.further_subdivide()
        return self.mesh


    def canonic_representation(self):
        K = np.array(self.mesh).reshape(-1,2)
        K = [tuple(x) for x in K]
        N = len(set(K))
        self.v_indexes = [range(N)]
        self.mapping = list(set(K))
        for tri in self.mesh:
            self.triangles.append([self.mapping.index(tuple(pt)) for pt in tri])


    def area_3D(self,tr):
        v = [self.mapping[tr[0]],self.mapping[tr[1]],self.mapping[tr[2]]]
        return area_3D(v)


    def N(self,i):
        N = []
        for tr in self.triangles :
            if i in tr:
                N += list(tr)
        return list(set(N))

    def S(self,i,j):
        S = 0
        for tr in self.triangles:
            if (i in tr) and (j in tr):
                S += area_3D(tr)
        return S

    def calcul_weights(self,i,j,k) : 





        


    
    
    












