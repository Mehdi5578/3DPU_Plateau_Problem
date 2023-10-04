from PointList import *
from Triangle import *
from tqdm import tqdm
from copy import *


class TriangularMesh:
    def __init__(self, boundary : PointList, desired_triangle_count):
        self.boundary = boundary 
        self.m = desired_triangle_count
        self.mesh = []
        self.n = len(boundary.points)
        self.v_indexes = [] #the vertexes indices
        self.mapping = [] #contains the mapping from I(v_indexes) to R^3
        self.triangles  = []  #each triangle is a tuple
        self.inside_indexes  = []
        self.N = {} #les indices des voisins 
        self.N_vertexes = len(self.v_indexes) # the number of nodes in the meshing
        self.w = np.empty((len(self.v_indexes),len(self.v_indexes)))
        self.dict_vertexes = {} # contains the dict of all triangles associated to an index
        self.edges = set()
    
    def add_points_to_boundary(self,N = 10):
        "adding points to the boundary"
        K = deepcopy(self.boundary.points)
        R = []
        for i in tqdm(range(len(self.boundary.points))):
            
            p1 = K[i-1]
            p2 = K[i]
            L = list(np.linspace(p1,p2,N+2))[1:-1]

            R.append(p1)
            R = R + L
        self.boundary.points = R
        self.n = len(self.boundary.points)



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
        self.further_subdivide()
        return self.mesh


    def modify_N(self,i):
        N = []
        for tr in self.triangles :
            if i in tr:
                N += list(tr)
        self.N[i] = list(set(N))


    def canonic_representation_from_mesh(self):
        K = np.array(self.mesh).reshape(-1,3)
        K = [tuple(x) for x in K]
        N = len(set(K))
        self.v_indexes = list(range(N))
        self.mapping = list(set(K))
        for j in self.v_indexes:
            self.dict_vertexes[j] = []
        self.triangles = []
        for tri in self.mesh:
            triangle = [self.mapping.index(tuple(pt)) for pt in tri]
            self.triangles.append(triangle)
            self.dict_vertexes[triangle[0]].append(tuple(triangle))
            self.dict_vertexes[triangle[1]].append(tuple(triangle))
            self.dict_vertexes[triangle[2]].append(tuple(triangle))

        for i in self.v_indexes:
            self.modify_N(i)
    
    def tuple_mapping(self):
        self.mapping = [tuple(k) for k in self.mapping]

            


    def area_3D(self,tr):
        v = [self.mapping[tr[0]],self.mapping[tr[1]],self.mapping[tr[2]]]
        return area_3D(v)


    def _N(self,i):
        N = []
        for tr in self.triangles :
            if i in tr:
                N += list(tr)
        return list(set(N))



    def S(self,i,j):
        S = 0
        tr_i = set(self.dict_vertexes[i])
        tr_j = set(self.dict_vertexes[j])
        intersect = list(tr_i.intersection(tr_j))
        for tr in intersect :
            S+= self.area_3D(tr)
        return S


    def calcul_weights(self,i,j) :
        S = 0
        for k in self.N[i]:
            S += self.S(i,k)
        self.w[i,j] =  self.S(i,j)/S




    def cotangent_angle(self, p1, p2, p3):
        """Compute the cotangent of the angle between p1-p2 and p1-p3."""
        v1 = np.array(self.mapping[p2]) - np.array(self.mapping[p1])
        v2 = np.array(self.mapping[p3]) - np.array(self.mapping[p1])
        dot_product = np.dot(v1, v2)
        cross_product_norm = np.linalg.norm(np.cross(v1, v2))
        return dot_product / cross_product_norm

    def adjacent_area(self, vertex):
        """Compute the total area of triangles adjacent to the vertex."""
        area = 0
    
        triangles = self.dict_vertexes[vertex]
        for tri in triangles:
            area += self.area_3D(tri)
    
        return area


    def compute_mean_curvature(self):
        """Compute the mean curvature for each vertex in the mesh."""
        vertex_curvatures = dict()
        for i in self.inside_indexes:
            A_i = self.adjacent_area(i)
            curvature_sum = np.zeros(3)
            
            for j in self.N[i]:
                tr_i = set(self.dict_vertexes[i])
                tr_j = set(self.dict_vertexes[j])
                t1,t2 = list(tr_i.intersection(tr_j))

                alpha = [r for r in t1 if r not in (i,j)][0]
                beta = [r for r in t2 if r not in (i,j)][0]

                cot_alpha = self.cotangent_angle(alpha,i,j)
                cot_beta = self.cotangent_angle(beta,i,j)

                p_i = np.array(self.mapping[i]) 
                p_j = np.array(self.mapping[j]) 

                curvature_sum += (cot_alpha + cot_beta) *(p_j - p_i)

            
            h_i = np.linalg.norm(curvature_sum) / (4 * A_i)
            vertex_curvatures[i] = h_i
        return vertex_curvatures



        


    
    
    












