from PointList import *



def area_3D(vertices):
    # Compute the area of the triangle 
    v1 = [vertices[1][i] - vertices[0][i] for i in range(3)]
    v2 = [vertices[2][i] - vertices[0][i] for i in range(3)]
    cross_product = [
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ]
    return 0.5 * sum(x**2 for x in cross_product)**0.5

def can_flip(edge, mesh):
    triangles = [triangle for triangle in mesh if set(edge).issubset(set(triangle.vertices))]
    return len(triangles) == 2

def flip_edge(edge, mesh):
    triangles = [triangle for triangle in mesh if set(edge).issubset(set(triangle.vertices))]
    t1, t2 = triangles
    opposite_vertices = [v for v in t1.vertices if v not in edge] + [v for v in t2.vertices if v not in edge]
    new_triangles = [Triangle(edge[0], opposite_vertices[0], opposite_vertices[1]), 
                     Triangle(edge[1], opposite_vertices[0], opposite_vertices[1])]
    for triangle in triangles:
        mesh.remove(triangle)
    mesh.extend(new_triangles)

def lawson_flip(mesh):
    edges = set()
    for triangle in mesh:
        for i in range(3):
            edge = tuple(sorted([triangle.vertices[i], triangle.vertices[(i+1)%3]]))
            edges.add(edge)
    for edge in edges:
        if can_flip(edge, mesh):
            triangles = [triangle for triangle in mesh if set(edge).issubset(set(triangle.vertices))]
            old_area = sum(triangle.area() for triangle in triangles)
            flip_edge(edge, mesh)
            new_area = sum(triangle.area() for triangle in triangles)
            if new_area >= old_area:
                flip_edge(edge, mesh)  # Flip back if the new area is not smaller
