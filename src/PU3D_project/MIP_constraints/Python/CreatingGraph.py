import numpy as np

class GraphGrid3D:

    def __init__(self,Edges,Marked_edges):
        # Constructor code here
        
        self.x_min = min(min(edge[1][0],edge[0][0]) for edge in Edges) - 1
        self.x_max = max(max(edge[1][0],edge[0][0]) for edge in Edges) + 1
        self.y_min = min(min(edge[1][1],edge[0][1]) for edge in Edges) - 1
        self.y_max = max(max(edge[1][1],edge[0][1]) for edge in Edges) + 1
        self.z_min = min(min(edge[1][2],edge[0][2]) for edge in Edges) - 1
        self.z_max = max(max(edge[1][2],edge[0][2]) for edge in Edges) + 1
        self.mapping = [] 
        self.index_mapping = {}
        self.edges = {}
        self.marked_edges = Marked_edges
        self.marked_edges_index = {}
        self.marked = {}
        self.Graph = {}
        self.Graph_edges = {}

        self.blockedges = []
        for edge in Edges:
            if edge not in Marked_edges:
                self.blockedges.append(edge)
        
        self.blocked_edges = set()
        self.edges = set()
        self.map_edges = []
        self.map_edges_index = {}
        self.fill_mapping()
        self.create_graph()
        self.create_edges()
        self.refill_edges()
        self.index_blocked_edges = set([self.map_edges_index[edge] for edge in self.blocked_edges])
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
            edge = (min(index_i,index_j),max(index_i,index_j))
            if edge in self.edges:
                self.blocked_edges.add(edge)

   
    def get_neighbors(self,point):
        i,j,k = point
        neighbors = set()
        if i < self.x_max :
            neighbors.add(self.index_mapping[(i+1,j,k)])
        if i > self.x_min:
            neighbors.add(self.index_mapping[(i-1,j,k)])
        if j < self.y_max:
            neighbors.add(self.index_mapping[(i,j+1,k)])
        if j > self.y_min:
            neighbors.add(self.index_mapping[(i,j-1,k)])
        if k < self.z_max :
            neighbors.add(self.index_mapping[(i,j,k+1)])
        if k > self.z_min:
            neighbors.add(self.index_mapping[(i,j,k-1)])
        return neighbors


    def create_graph(self):
        for point in self.mapping:
            self.Graph[self.index_mapping[point]] = set()
        for point in self.mapping:
            self.Graph[self.index_mapping[point]] = self.get_neighbors(point)
        for edge in self.marked_edges:
            i,j = edge
            index_i,index_j = self.index_mapping[tuple(i)],self.index_mapping[tuple(j)]
            self.Graph[index_i].remove(index_j)
            self.Graph[index_j].remove(index_i)
        
    
    def create_edges(self):
        for point in self.mapping:
            for next_point in self.Graph[self.index_mapping[point]]:
                index_point = self.index_mapping[point]
                self.edges.add((min(index_point,next_point),max(index_point,next_point)))
        self.map_edges = list(self.edges)
        for i in range(len(self.map_edges)):
            self.map_edges_index[self.map_edges[i]] = i
    

    def fill_Graph_edges(self):
        for edge in self.map_edges:
            self.Graph_edges[edge] = []
        for edge in self.map_edges:
            i,j = edge
            neighbours_i = self.Graph[i]
            for neigh_i in neighbours_i:
                if neigh_i != j:
                    self.Graph_edges[edge].append((min(neigh_i,i),max(neigh_i,i)))
            neighbours_j = self.Graph[j]
            for neigh_j in neighbours_j:
                if neigh_j != i:
                    assert((min(neigh_j,j),max(neigh_j,j)) in self.map_edges),(min(neigh_j,j),max(neigh_j,j)) 
                    self.Graph_edges[edge].append((min(neigh_j,j),max(neigh_j,j)))


    def detect_cycle(self,graph,origin):
        visited = [False] * len(graph)
        node = origin
        queue = [origin]
        father = {origin:origin}
        path = {origin:[]}
        number_of_edges = {origin:[]}
        while queue:
            node = queue.pop(0)
            for next_node in [next_node for next_node in graph[node] if next_node != father[node] and not visited[next_node]]:
                father[next_node] = node
                path[next_node] = path[node]+[node]
                if (min(node,next_node),max(node,next_node)) in self.blocked_edges:
                    number_of_edges[next_node] = number_of_edges[node] + [(min(node,next_node),max(node,next_node))]
                else:
                    number_of_edges[next_node] = number_of_edges[node]
                queue.append(next_node)
                visited[next_node] = True
        return path, number_of_edges


    def merge(self,path_i,path_j):
        cpt = 0
        while path_i[cpt] == path_j[cpt]:
            cpt += 1
        return [path_i[cpt-1]] + path_i[cpt:] + path_j[cpt:][::-1]


    def detect_impair_cycle(self,origin):
        #ajouter des conditions pour les cycles impairs
        path,number_of_edges = self.detect_cycle(self.Graph,origin)
        cycles = {}
        for edge in self.map_edges:
            i,j = edge
            nbre_blocked_edges = len(number_of_edges[i]) + len(number_of_edges[j]) 
            if nbre_blocked_edges % 2 == 1 and (min(i,j),max(i,j)) not in self.blocked_edges:
                if nbre_blocked_edges not in cycles:
                    cycles[nbre_blocked_edges] = []
                else:
                    cycles[nbre_blocked_edges].append(self.merge(path[i]+[i],path[j]+[j]))
            if nbre_blocked_edges % 2 == 0 and (min(i,j),max(i,j)) in self.blocked_edges:
                if nbre_blocked_edges+1 not in cycles:
                    cycles[nbre_blocked_edges+1] = []
                else:
                    cycles[nbre_blocked_edges+1].append(self.merge(path[i]+[i],path[j]+[j]))
        return cycles


def main():
    # Create an instance of GraphGrid3D
    graph = GraphGrid3D(0, 10, 20, 0, 5, 15,[])

    # Fill the mapping
    graph.fill_mapping()

    # Create the graph
    graph.create_graph()

if __name__ == "__main__":
    main()
