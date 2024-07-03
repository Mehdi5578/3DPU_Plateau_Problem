import numpy as np

class PointList:
    def __init__(self):
        self.points = []

    def add_point(self, point):
        self.points.append(np.array(point))

    def sum_points(self):
        total = np.sum(self.points, axis=0)
        return total

    def average_point(self):
        avg = np.mean(self.points, axis=0)
        return avg

    def display_points(self):
        for point in self.points:
            print(point)
