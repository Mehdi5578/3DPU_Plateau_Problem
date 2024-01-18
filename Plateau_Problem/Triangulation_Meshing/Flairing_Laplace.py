from Plateau_Problem.Triangulation_Meshing.Initial_triangulation import TriangularMesh
from Plateau_Problem.Triangulation_Meshing.PointList import *
from tqdm import tqdm
import math




class Updating_Laplace(TriangularMesh):
    def __init__(self, boundary : PointList, desired_triangle_count):
        super().__init__(boundary, desired_triangle_count)
        self.S

    def S(self,i,j):
        S = 0
        intersect = self.common_dict_vertexes[tuple(sorted((i,j)))]
        assert len(intersect) == 2, "Sorry, it should have only two adjacent triangles but" + str(i) +" "+ str(j) + "has" + str(len(intersect)) + "adjacent triangles"
        for tr in intersect :
            S+= self.area_3D(tr)
        return S


    def calcul_weights(self,i,j) :
        S_ij = self.S(i, j)
        S = sum(self.S(i, k) for k in self.N[i])
        self.w[i,j] =  S_ij / S


         


    def update_weights(self):
        "Updates the weights of the Mesh for each iteration"
        for j in tqdm(self.inside_indexes):
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

        for k in N_i:
            self.calcul_weights(i,k)

            
    
        




