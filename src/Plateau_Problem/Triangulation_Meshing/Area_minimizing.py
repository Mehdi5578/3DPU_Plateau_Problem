from Plateau_Problem.Triangulation_Meshing.Initial_triangulation import TriangularMesh
from Plateau_Problem.Triangulation_Meshing.PointList import *
from tqdm import tqdm
from Plateau_Problem.Triangulation_Meshing.Flairing_Laplace import *
from Plateau_Problem.Triangulation_Meshing.Final_surface import *

class Final_minimization(Edge_Flipping):
    def __init__(self, boundary : PointList, desired_triangle_count):
        super().__init__(boundary, desired_triangle_count)

    def C_inversed(self,h):
        "Caomputes the inversed C matrix of index h"
        identity_matrix = np.identity(3)
        C = 0

        for tr in self.dict_vertexes[h]:
            j,k = tuple([a for a in tr if a != h ])
            P_j = np.array(self.mapping[j])
            P_k = np.array(self.mapping[k])
            P_h = np.array(self.mapping[h])

            # Assuming P_j * P_k is element-wise multiplication
            product_jk = P_j * P_k

            # Numerator part
            numerator = np.dot(product_jk**2, identity_matrix) - np.outer(product_jk, product_jk.T)

            # Cross product for denominator
            cross_product = np.cross(product_jk, P_j * P_h)

            # Update C with the contribution from this triplet
            C += numerator / np.sqrt(np.dot(cross_product, cross_product)**2)

        return np.linalg.inv(C)
    
    def update_mapping_area(self,h):
        "Update the position of h to minimze the said area"
        P_bar_h = np.zeros(3)
        for tr in self.dict_vertexes[h]:
            j,k = tuple([a for a in tr if a != h ])
            P_j = np.array(self.mapping[j])
            P_k = np.array(self.mapping[k])
            P_h = np.array(self.mapping[h])

            product_jk = P_j * P_k  # Element-wise multiplication
            dot_product_jk_j = np.dot(product_jk, P_j)  # Dot product
            cross_product_jk_jh = np.cross(product_jk, P_j * P_h)  # Cross product

            # The term inside the summation
            term = ((dot_product_jk_j * product_jk) - (product_jk**2 * P_j)) / np.sqrt(np.dot(cross_product_jk_jh, cross_product_jk_jh)**2)

            # Summation
            P_bar_h += term

        # Multiply with the inverse of C
        P_bar_h = -np.dot(self.C_inversed(h), P_bar_h)

    


