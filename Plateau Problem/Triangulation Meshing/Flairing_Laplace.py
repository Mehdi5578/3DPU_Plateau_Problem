from Initial_triangulation import TriangularMesh
from PointList import *
from tqdm import tqdm



class Updating_Laplace(TriangularMesh):
    def __init__(self, boundary : PointList, desired_triangle_count):
        super().__init__(boundary, desired_triangle_count)

    def update_weights(self):
        "Updates the weights of the Mesh for each iteration"
        for i in tqdm(range(self.N_vertexes)):
            for j in (range(self.N_vertexes)):
                if i in self.N[j] :
                    self.calcul_weights(i,j)

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
            P = P + self.w[i,k]*np.array(self.mapping[k])
        print(P)
        self.mapping[i] = P
    
    

    

        




