from src.PU3D_project.Plateau_Problem.Triangulation_Meshing.Initial_triangulation import TriangularMesh
from src.PU3D_project.Plateau_Problem.Triangulation_Meshing.PointList import *
from tqdm import tqdm
import math




class Updating_Laplace(TriangularMesh):
    
    def __init__(self, boundary : PointList, desired_triangle_count):
        super().__init__(boundary, desired_triangle_count)
        self.triangle_area_dict = {}
        self.S_dict = {}
        
    # create dict of triangles and their area
    def fill_area_dict(self):
        for triangle in self.triangles:
            self.triangle_area_dict[tuple(triangle)] = self.area_3D(triangle)
        for triangle in self.triangles:
            self.S_dict[tuple(sorted((triangle[0],triangle[1])))] = self.S(triangle[0],triangle[1])
            self.S_dict[tuple(sorted((triangle[0],triangle[2])))] = self.S(triangle[0],triangle[2])
            self.S_dict[tuple(sorted((triangle[1],triangle[2])))] = self.S(triangle[1],triangle[2])

    def area_3D(self,triangle):
        "Calculates the area of a triangle in 3D"
        A = self.mapping[triangle[0]]
        B = self.mapping[triangle[1]]
        C = self.mapping[triangle[2]]
        AB = [B[0] - A[0], B[1] - A[1], B[2] - A[2]]
        AC = [C[0] - A[0], C[1] - A[1], C[2] - A[2]]

        # Calculate the cross product of AB and AC
        cross_product = [
            AB[1]*AC[2] - AB[2]*AC[1],
            AB[2]*AC[0] - AB[0]*AC[2],
            AB[0]*AC[1] - AB[1]*AC[0]
        ]

        # Calculate the norm of the cross product (magnitude)
        norm = (cross_product[0]**2 + cross_product[1]**2 + cross_product[2]**2)**0.5

        # Area of the triangle is half the magnitude of the cross product
        return 0.5 * norm
    
    
    def S(self,i,j):
        S = 0
        intersect = self.common_dict_vertexes[(min(i,j),max(i,j))]
        # assert len(intersect) == 2, "Sorry, it should have only two adjacent triangles but" + str(i) +" "+ str(j) + "has" + str(len(intersect)) + "adjacent triangles"
        for tr in intersect :
            S+= self.triangle_area_dict[tuple(tr)]
        return S


    def calcul_weights(self,i,j) :
        S_ij = self.S_dict[(min(i,j),max(i,j))]
        S = sum(self.S_dict[(min(i,k),max(i,k))] for k in self.N[i])
        self.w[i,j] =  S_ij / S


         


    def update_weights(self):
        "Updates the weights of the Mesh for each iteration"
        self.fill_area_dict()
        for j in (self.inside_indexes):
            for i in self.N[j] :
                self.calcul_weights(j,i)
                

    def calculate_area(self):
        "Calculates the area of the meshing"
        S = 0
        for tri in self.triangles:
            S = S + self.area_3D(tri)
        return S

    def update_mapping(self,i):
        "Updates the position of the nodes using the Laplace Fairing"
        N_i = self.N[i]
        P = np.array([0,0,0])
        for k in N_i :
            P = P + (self.w[i,k])*np.array(self.mapping[k])
        
        self.mapping[i] = P
        # for tr in self.dict_vertexes[i]:
        #     self.triangle_area_dict[tr] = self.area_3D(tr)
        # for k in N_i:
        #     self.S_dict[(min(i,k),max(i,k))] = self.S(i,k)
            
        # for k in N_i:
        #     self.calcul_weights(i,k)
        #     if k not in self.outside_vertexes:
        #         self.calcul_weights(k,i)
            

            
    
        




