from src.PU3D_project.Plateau_Problem.Triangulation_Meshing.PointList import *
from src.PU3D_project.Plateau_Problem.Triangulation_Meshing.Triangle import *
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
        self.outside_vertexes = set()
        self.N = {} #les indices des voisins 
        self.N_vertexes = len(self.v_indexes) # the number of nodes in the meshing
        self.w = np.empty((len(self.v_indexes),len(self.v_indexes)))
        self.dict_vertexes = {} # contains the dict of all triangles associated to an index
        self.edges = set()
        self.common_dict_vertexes = {}
        self.vertex_curvatures = dict()
        self.mapping_index = {}
        self.position_triangle = dict()
    

    def add_points_to_boundary(self,N = 10):
        "adding points to the boundary"
        K = deepcopy(self.boundary.points)
        R = []
        for i in (range(len(self.boundary.points))):
            
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
        P = self.create_initial_subdivisions()
        s = int(self.m/(2*self.n) - 1/2)
        for j in (range(s)) : 
            for i in range(self.n) :
                self.mesh.append([P[j,i],P[j+1,i+1],P[j,i+1]])
                self.mesh.append([P[j,i],P[j+1,i+1],P[j+1,i]])
        
        


    def split_quadrilateral(self):
        C = self.compute_central_point()
        P = self.create_initial_subdivisions()
        for i in (range(self.n)) :
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
        # self.further_subdivide()
        return self.mesh


    def modify_N(self,i):
        N = set()
        for tr in self.dict_vertexes[i] :
            for j in tr :
                if j != i :
                    N.add(j)
        self.N[i] = N



    def canonic_representation_from_mesh(self):
        K = np.array(self.mesh).reshape(-1,3)
        K = [tuple(x) for x in K]
        N = len(set(K))
        self.N_vertexes = N
        self.w = np.empty((N,N))
        self.v_indexes = list(range(N))
        self.mapping = list(set(K))
        cpt = 0
        for point in self.mapping:
            self.mapping_index[point] = cpt
            cpt += 1

        for j in self.v_indexes:
            self.dict_vertexes[j] = set()
        
        self.triangles = []



        for tr in (self.mesh):
            tri = [self.mapping_index[tuple(pt)] for pt in tr]
            if len(set(tri)) == 3:
                self.common_dict_vertexes[tuple(sorted((tri[0],tri[1])))] = set()
                self.common_dict_vertexes[tuple(sorted((tri[0],tri[2])))] = set()
                self.common_dict_vertexes[tuple(sorted((tri[1],tri[2])))] = set() 
        cpt = 0
        for tri in (self.mesh):
            triangle = [self.mapping_index[tuple(pt)] for pt in tri]
            tr = tuple(sorted(triangle))
            if len(set(tr)) == 3:
                self.triangles.append(tr)
                self.position_triangle[tr] = cpt
                cpt = cpt + 1
                self.dict_vertexes[triangle[0]].add(tuple(sorted(triangle)))
                self.dict_vertexes[triangle[1]].add(tuple(sorted(triangle)))
                self.dict_vertexes[triangle[2]].add(tuple(sorted(triangle)))
                self.common_dict_vertexes[tuple(sorted((triangle[0],triangle[1])))].add(tuple(sorted(triangle)))
                self.common_dict_vertexes[tuple(sorted((triangle[0],triangle[2])))].add(tuple(sorted(triangle)))
                self.common_dict_vertexes[tuple(sorted((triangle[1],triangle[2])))].add(tuple(sorted(triangle)))

        for i in (self.v_indexes):
            self.modify_N(i)

        Outside = [tuple(k) for k in self.boundary.points]
        self.outside_vertexes = set([self.mapping_index[pt] for pt in Outside])
        self.inside_indexes = list(set(self.v_indexes) - self.outside_vertexes)

    
    def tuple_mapping(self):
        self.mapping = [tuple(k) for k in self.mapping]

            
    def clean_triangles(self):
        self.triangles

    def area_3D(self,tr):
        v = [self.mapping[tr[0]],self.mapping[tr[1]],self.mapping[tr[2]]]
        return area_3D(v)


    def _N(self,i):
        N = []
        for tr in self.triangles :
            if i in tr:
                N += list(tr)
        return list(set(N))


    def cross(self,a,b):
        a1,a2,a3 = a
        b1,b2,b3 = b
        return np.array([a2*b3 - a3*b2 , a3*b1 - a1*b3 , a1*b2 - a2*b1])

    def dot(self,a,b):
        a1,a2,a3 = a
        b1,b2,b3 = b
        return a1*b1 + a2*b2 + a3*b3

    

    def cotangent_angle(self, p1, p2, p3):
        """Compute the cotangent of the angle between p1-p2 and p1-p3."""
        v1 = (self.mapping[p2]) - (self.mapping[p1])
        v2 = (self.mapping[p3]) - (self.mapping[p1])
        dot_product = self.dot(v1, v2)
        cross_prodcut = self.cross(v1, v2)
        cross_product_norm = (cross_prodcut[0]**2 + cross_prodcut[1]**2 + cross_prodcut[2]**2)**0.5
        return dot_product / cross_product_norm

    def voronoi(self,p1,p2,p3):
        """Compute vornoi area at point p"""

        p = (self.mapping[p1])
        q = (self.mapping[p2])
        r = (self.mapping[p3])
        pr = np.linalg.norm(p-r)**2
        pq = np.linalg.norm(p-q)**2
        cot_q = self.cotangent_angle(p2,p3,p1)
        cot_r = self.cotangent_angle(p3,p1,p2)
    
        return (pr*cot_q + pq*cot_r)/8
    

    def is_obtuse(self,p1, p2,p3):
        v1 = self.mapping[p1]
        v2 = self.mapping[p2]
        v3 = self.mapping[p3]
        # Compute squared lengths of the sides
        a2 = (v2[0]-v3[0])**2 + (v2[1]-v3[1])**2 + (v2[2]-v3[2])**2
        b2 = (v1[0]-v3[0])**2 + (v1[1]-v3[1])**2 + (v1[2]-v3[2])**2
        c2 = (v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2

        # Check if any angle is obtuse
        return a2 > b2 + c2 or b2 > a2 + c2 or c2 > a2 + b2


    def obtuse_at_point(self,p1, p2,p3):
        v1 = self.mapping[p1]
        v2 = self.mapping[p2]
        v3 = self.mapping[p3]
        # Compute squared lengths of the sides
        a2 = (v2[0]-v3[0])**2 + (v2[1]-v3[1])**2 + (v2[2]-v3[2])**2
        b2 = (v1[0]-v3[0])**2 + (v1[1]-v3[1])**2 + (v1[2]-v3[2])**2
        c2 = (v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2

        # Check if any angle is obtuse
        return a2 > b2 + c2 
    

    def mixed_area(self,v1,v2,v3):
        " calculate the voronoi mixed area at v1 "
        if not self.is_obtuse(v1,v2,v3): # checks that the triangle is not obtuse
            return self.voronoi(v1,v2,v3)
        else:
            tr = (v1,v2,v3)
            if self.obtuse_at_point(v1,v2,v3):
                
                return self.area_3D(tr)/2
            else:
                return self.area_3D(tr)/4

    def voronoi_area(self, vertex):
        """Compute the total area of triangles adjacent to the vertex."""
        area = 0
    
        triangles = self.dict_vertexes[vertex]
        for tri in triangles:
            v1 = vertex
            assert (len([a for a in tri if a != vertex]) == 2), vertex
            v2 = [a for a in tri if a != vertex][0]
            v3 = [a for a in tri if a != vertex][1]
            area += self.mixed_area(v1,v2,v3)
        return area


    def compute_mean_curvature(self):
        """Compute the mean curvature for each vertex in the mesh."""
        self.vertex_curvatures = dict()
        for i in self.inside_indexes:

            A_i = self.voronoi_area(i) #We do it with simple area
            curvature_sum = np.zeros(3)
            B_i  = 0
            for j in self.N[i]:
                
                t1,t2 = self.common_dict_vertexes[(min(i,j),max(i,j))]
                

                alpha = [r for r in t1 if r not in (i,j)][0]
                beta = [r for r in t2 if r not in (i,j)][0]

                cot_alpha = self.cotangent_angle(alpha,i,j)
                cot_beta = self.cotangent_angle(beta,j,i)

                p_i = np.array(self.mapping[i]) 
                p_j = np.array(self.mapping[j]) 

                curvature_sum += (cot_alpha + cot_beta) *(p_j - p_i)
                
                if A_i < 0:
                    print(A_i,i)
                    raise Exception("Error this area si negative !!!")
                    

            
            h_i = np.linalg.norm(curvature_sum) / (4 * A_i)
            self.vertex_curvatures[h_i] = i
        A = np.array(list(self.vertex_curvatures.keys()))
        
        return np.max(A)

