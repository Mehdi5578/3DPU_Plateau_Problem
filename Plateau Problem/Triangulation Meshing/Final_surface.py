from Flairing_Laplace import *
from copy import *

class Edge_Flipping(Updating_Laplace):
    def __init__(self, boundary : PointList, desired_triangle_count):
        super().__init__(boundary, desired_triangle_count)

    def fill_edges(self):
        for tri in self.triangles:
            for i in range(3):
                edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
                self.edges.add(edge)

    def test(self,edge,tri):
        return set(edge).issubset(set(tri))
    
    def can_flip(self,edge):
        triangles = [triangle for triangle in self.triangles if self.test(edge,triangle)]
        return len(triangles) == 2
    
    def flip_edge(self,edge):
        triangles = [triangle for triangle in self.triangles if self.test(edge,triangle)]
        t1,t2 = triangles
        opposite_vertices = [v for v in t1 if v not in edge] + [v for v in t2 if v not in edge]
        new_triangles = [[edge[0], opposite_vertices[0], opposite_vertices[1]], 
                     [edge[1], opposite_vertices[0], opposite_vertices[1]]]
        for triangle in triangles:
            self.triangles.remove(triangle)
        self.triangles.extend(new_triangles)
        return triangles,new_triangles
    
    def lawson_flip(self):
       
        for edge in self.edges:
            if self.can_flip(edge):
                triangles = [triangle for triangle in self.triangles if self.test(edge,triangle)]
                old_area = sum(self.area_3D(triangle) for triangle in triangles)
                old_tr, new_tr = self.flip_edge(edge)
                new_area = sum(self.area_3D(triangle) for triangle in triangles)

                if new_area >=  old_area: 
                    for triangle in new_tr:
                        self.triangles.remove(triangle)
                    self.triangles.extend(old_tr) # Flip back if the new area is not smaller


        

        


