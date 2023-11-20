from Plateau_Problem.Triangulation_Meshing.Flairing_Laplace import *
import copy

class Edge_Flipping(Updating_Laplace):
    def __init__(self, boundary : PointList, desired_triangle_count):
        super().__init__(boundary, desired_triangle_count)
        self.dict_edges = dict()
        self.L = {}
        self.L_replaced = {}

    def fill_edges(self):
        self.edges = set()
        for tri in self.triangles:
            for i in range(3):
                edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
                self.edges.add(edge)
                self.dict_edges[edge] = -1

    def test(self,edge,tri):
        return set(edge).issubset(set(tri))
    
    def can_flip(self,edge):
        triangles = [tuple(set(sorted(triangle)))  for triangle in self.triangles if self.test(edge,triangle)]
        return len(set(triangles)) == 2
    
    def flip_edge(self,edge):

        old_triangles = [triangle for triangle in self.triangles if self.test(edge,triangle)]
        t1,t2 = old_triangles
        opposite_vertices = [v for v in t1 if v not in edge] + [v for v in t2 if v not in edge]
        new_triangles = [tuple(sorted([edge[0], opposite_vertices[0], opposite_vertices[1]])), 
                     tuple(sorted([edge[1], opposite_vertices[0], opposite_vertices[1]]))]
        return old_triangles,new_triangles
    
    def lawson_flip(self):

        swaped = True
        ll = len(self.edges)

        while swaped :
            
            swaped = False

            for edge in (self.edges):
                if self.can_flip(edge):

                    old_tr, new_tr = self.flip_edge(edge)
                    old_area = sum(self.area_3D(triangle) for triangle in old_tr)
                    new_area = sum(self.area_3D(triangle) for triangle in new_tr)
                    
                    R = set(np.array(new_tr).reshape(-1))
                    edge1 = tuple(sorted([a for a in R if a not in edge]))
                    
                    if new_area < old_area and edge1 not in self.edges :
                        # print("this edge.{} was swapped with this one {} resulting in an area minimization of {}".format(edge,edge1,old_area-new_area))
                        # D = D + new_area - old_area
                        # print("hat")
                        
                        swaped = True

                        self.edges.add(edge1)
                        self.edges.remove(edge)

                        for tr in new_tr:
                            self.triangles.append(tr)
                        
                        for tr in old_tr :
                            self.triangles.remove(tr)

                        for tr in new_tr:
                            for pt in tr:
                                self.dict_vertexes[pt].append(tuple(sorted(tr)))
                        
                        for tr in old_tr:
                            for pt in tr:
                                self.dict_vertexes[pt].remove(tuple(sorted(tr)))

                        
                        a,b = edge
                        a1,b1 = edge1

                        self.N[a1].append(b1)
                        self.N[b1].append(a1)

                        self.N[a].remove(b)
                        self.N[b].remove(a)  
                                              
                    self.L[edge] = (new_area -  old_area, old_tr, new_tr,edge)
                    if len(list(self.edges)) < ll:
                        print(edge,edge1)


                
    def C_inversed(self,h):
        "Computes the inversed C matrix of index h"
        identity_matrix = np.identity(3)
        C = 0

        for tr in self.dict_vertexes[h]:
            
            j,k = tuple([a for a in tr if a != h ])
            P_j = np.array(self.mapping[j])
            P_k = np.array(self.mapping[k])
            P_h = np.array(self.mapping[h])

            vect_jk = (P_k - P_j)
            vect_jh = (P_h - P_j)

            numerator = (np.linalg.norm(vect_jk)**2)*identity_matrix  - np.outer(vect_jk, vect_jk.T)
            cross_product = np.cross(vect_jk, vect_jh)
            C += numerator / np.linalg.norm(cross_product)
        return np.linalg.inv(C)
    
    def update_mapping_area(self,h):
        "Update the position of h to minimze the said area"
        P_bar_h = np.zeros(3)
        for tr in self.dict_vertexes[h]:
            j,k = tuple([a for a in tr if a != h ])
            P_j = np.array(self.mapping[j])
            P_k = np.array(self.mapping[k])
            P_h = np.array(self.mapping[h])

            vect_jk = P_k - P_j
            vect_jh = P_h - P_j

            dot_product_jk_j = np.dot(vect_jk, P_j)  # Dot product
            cross_product_jk_jh = np.cross(vect_jk, vect_jh)  # Cross product

            # The term inside the summation
            term = (np.dot(dot_product_jk_j,vect_jk) - np.dot(np.linalg.norm(vect_jk)**2 , P_j)) / np.linalg.norm(cross_product_jk_jh)

            # Summation
            P_bar_h += term

        # Multiply with the inverse of C
        P_bar_h = -np.dot(self.C_inversed(h), P_bar_h)
        self.mapping[h] =  P_bar_h


if __name__ == "__main__":
    # Code here will only run when the file is executed directly,
    # not when it is imported as a module.
    print("Executed when invoked directly")

