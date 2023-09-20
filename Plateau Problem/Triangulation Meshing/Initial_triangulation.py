from PointList import *

class TriangularMesh:
    def __init__(self, boundary, desired_triangle_count):
        self.boundary = boundary
        self.desired_triangle_count = desired_triangle_count
        self.mesh = []
        

    def compute_central_point(self):
        #  Compute the central point by averaging the vertices of the boundary
        pass
        

    def create_initial_subdivisions(self):
        # TODO: Create initial subdivisions using the central point
        pass

    def split_quadrilaterals(self):
        # TODO: Split any quadrilaterals into triangles
        pass

    def further_subdivide(self):
        # TODO: Further subdivide triangles if the count doesn't match the desired amount
        pass

    def generate_mesh(self):
        self.compute_central_point()
        self.create_initial_subdivisions()
        self.split_quadrilaterals()
        self.further_subdivide()
        return self.mesh


C = PointList.PointList()
# # Usage example:
# if __name__ == "__main__":
#     boundary = [(x1, y1), (x2, y2), ...]  # Define the boundary vertices here
#     desired_triangle_count = m  # Set the desired triangle count here
#     mesh_generator = TriangularMesh(boundary, desired_triangle_count)
#     mesh = mesh_generator.generate_mesh()
#     print(mesh)
