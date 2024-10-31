from src.PU3D_project.Plateau_Problem.Triangulation_Meshing.Initial_triangulation import TriangularMesh
from src.PU3D_project.Plateau_Problem.Triangulation_Meshing.PointList import *
from tqdm import tqdm
from src.PU3D_project.Plateau_Problem.Triangulation_Meshing.Flairing_Laplace import *
from src.PU3D_project.Plateau_Problem.Triangulation_Meshing.Final_surface import *
from src.PU3D_project.utils import *

def edge_intersects_triangle(edge_start, edge_end, A, B, C):

    EPSILON = 1e-8

    # Direction vector of the segment
    D = edge_end - edge_start

    # Triangle vertices
    V0 = A
    V1 = B
    V2 = C

    # Compute edges of the triangle
    edge1 = V1 - V0
    edge2 = V2 - V0

    # Begin calculating determinant - also used to calculate u parameter
    h = np.cross(D, edge2)
    a = np.dot(edge1, h)

    # If a is close to zero, the line segment is parallel to the triangle plane
    if -EPSILON < a < EPSILON:
        return False  # Parallel

    f = 1.0 / a
    s = edge_start - V0
    u = f * np.dot(s, h)

    # Check if the intersection lies outside the triangle
    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(D, q)

    # The intersection lies outside the triangle
    if v < 0.0 or u + v > 1.0:
        return False

    # Compute t to find out where the intersection point is on the line
    t = f * np.dot(edge2, q)

    # Check if the intersection point is on the segment
    if t < 0.0 or t > 1.0:
        return False  # The intersection point is not within the segment

    return True  # The edge intersects the triangle

def intersect_edges(tr):
    A,B,C = tr
# Given triangle vertices

    # Compute AABB
    min_coords = np.floor(np.minimum.reduce([A, B, C])).astype(int)
    max_coords = np.ceil(np.maximum.reduce([A, B, C])).astype(int)

    # Initialize list to store intersecting edges

    intersecting_edges = []

    # Iterate over grid cells within the bounding box
    for x in range(min_coords[0], max_coords[0] + 1):
        for y in range(min_coords[1], max_coords[1] + 1):
            for z in range(min_coords[2], max_coords[2] + 1):
                # Define the 12 edges of the grid cell
                # For simplicity, only a few edges are defined here
                cell_edges = [
                    (np.array([x, y, z]), np.array([x+1, y, z])),
                    (np.array([x, y, z]), np.array([x, y+1, z])),
                    (np.array([x, y, z]), np.array([x, y, z+1])),
                    # Add other edges accordingly
                ]
                # Check intersection with the triangle
                for edge_start, edge_end in cell_edges:
                    if edge_intersects_triangle(edge_start, edge_end, A, B, C):
                        intersecting_edges.append((edge_start, edge_end))
    return intersecting_edges




class Final_minimization(Edge_Flipping):
    def __init__(self, boundary : PointList, desired_triangle_count):
        super().__init__(boundary, desired_triangle_count)
        self.Edges = set()

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
    
    def find_traingles(self):
        List_triangles = list(self.triangles)
        array_triangles = []
        for tr in List_triangles:
            new_tr = [np.array(self.mapping[t]) for t in tr]
            array_triangles.append(new_tr)
        blocked_edges = []
        for tr in (array_triangles):
            blocked_edges.append(intersect_edges(tr))
        applatir_blocked_edges = [item for sublist in blocked_edges for item in sublist]
        applatir_blocked_edges = [(tuple(p1),tuple(p2)) for p1,p2 in applatir_blocked_edges]
        self.Edges = set(applatir_blocked_edges)

    


