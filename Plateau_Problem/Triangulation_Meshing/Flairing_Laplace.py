from Plateau_Problem.Triangulation_Meshing.Initial_triangulation import TriangularMesh
from Plateau_Problem.Triangulation_Meshing.PointList import *
from tqdm import tqdm
import math




class Updating_Laplace(TriangularMesh):
    def __init__(self, boundary : PointList, desired_triangle_count):
        super().__init__(boundary, desired_triangle_count)

    def S(self,i,j):
        S = 0
        tr_i = set(self.dict_vertexes[i])
        tr_j = set(self.dict_vertexes[j])
        intersect = list(tr_i.intersection(tr_j))
        if len(intersect) != 2:
            print(intersect)
            raise Exception("Sorry, it should have only two adjacent triangles")
        for tr in intersect :
            S+= self.area_3D(tr)
        return S


    def calcul_weights(self,i,j) :
        S = 0
        for k in self.N[i]:
            S += (self.S(i,k))
        
        self.w[i,j] =  (self.S(i,j))/S
        if math.isnan(self.w[i,j]):
            print(i,j)
            raise ValueError("S is nega tive")


         


    def update_weights(self):
        "Updates the weights of the Mesh for each iteration"
        for j in self.inside_indexes:
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

            
    
        




