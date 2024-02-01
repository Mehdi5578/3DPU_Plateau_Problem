import numpy as np
import matplotlib.pyplot as plt

class GraphGrid3D:

    def __init__(self,Edges):
        # Constructor code here
        
        self.x_min = min(min(edge[1][0],edge[0][0]) for edge in Edges)
        self.x_max = max(max(edge[1][0],edge[0][0]) for edge in Edges)
        self.y_min = min(min(edge[1][1],edge[0][1]) for edge in Edges)
        self.y_max = max(max(edge[1][1],edge[0][1]) for edge in Edges)
        self.z_min = min(min(edge[1][2],edge[0][2]) for edge in Edges)
        self.z_max = max(max(edge[1][2],edge[0][2]) for edge in Edges)
        self.mapping = []
        self.index_mapping = {}
        self.edges = {}
        self.marked = {}
        self.Graph = {}
        self.blockedges = Edges
        self.blocked_edges = set()
        self.edges = set()
        self.map_edges = []
        self.map_edges_index = {}
        self.fill_mapping()
        self.create_graph()
        self.create_edges()
        self.refill_edges()
        self.index_blocked_edges = [self.map_edges_index[edge] for edge in self.blocked_edges]
        self.index_edges = [self.map_edges_index[edge] for edge in self.edges]

    def fill_mapping(self):
        cpt = 0
        for i in range(self.x_min,self.x_max+1):
            for j in range(self.y_min,self.y_max+1):
                for k in range(self.z_min,self.z_max+1):
                    self.mapping.append((i,j,k))
                    self.index_mapping[(i,j,k)] = cpt
                    cpt += 1

    def refill_edges(self):
        for edge in self.blockedges:
            i,j = edge
            index_i,index_j = self.index_mapping[tuple(i)],self.index_mapping[tuple(j)]
            self.blocked_edges.add((min(index_i,index_j),max(index_i,index_j)))

    def get_neighbors(self,point):
        i,j,k = point
        neighbors = []
        if i < self.x_max - 1:
            neighbors.append(self.index_mapping[(i+1,j,k)])
        if i > self.x_min:
            neighbors.append(self.index_mapping[(i-1,j,k)])
        if j < self.y_max - 1:
            neighbors.append(self.index_mapping[(i,j+1,k)])
        if j > self.y_min:
            neighbors.append(self.index_mapping[(i,j-1,k)])
        if k < self.z_max - 1:
            neighbors.append(self.index_mapping[(i,j,k+1)])
        if k > self.z_min:
            neighbors.append(self.index_mapping[(i,j,k-1)])
        return neighbors

    def create_graph(self):
        for point in self.mapping:
            self.Graph[self.index_mapping[point]] = []
        for point in self.mapping:
            self.Graph[self.index_mapping[point]] = self.get_neighbors(point)
    
    def create_edges(self):
        for point in self.mapping:
            for next_point in self.Graph[self.index_mapping[point]]:
                index_point = self.index_mapping[point]
                self.edges.add((min(index_point,next_point),max(index_point,next_point)))
        self.map_edges = list(self.edges)
        for i in range(len(self.map_edges)):
            self.map_edges_index[self.map_edges[i]] = i
        
            

def main():
    # Create an instance of GraphGrid3D
    graph = GraphGrid3D(0, 10, 20, 0, 5, 15,[])

    # Fill the mapping
    graph.fill_mapping()

    # Create the graph
    graph.create_graph()

if __name__ == "__main__":
    main()
