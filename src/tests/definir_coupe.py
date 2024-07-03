import numpy as np
import networkx as nx
from tqdm import tqdm
import math
import sys
from .CreatingCycles import *

class simple_3D_Graph:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = x_min	
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

        self.graph = nx.Graph()
        self.mapping = [] 
        self.index_mapping = {} 
        self.edges = set()
        
        self.create_graph()
    
    def fill_points(self):
        for i in range(self.x_min,self.x_max + 1):
            for j in range(self.y_min,self.y_max + 1):
                for k in range(self.z_min,self.z_max + 1):
                    self.mapping.append((i,j,k))
                    self.index_mapping[(i,j,k)] = len(self.mapping)-1
    
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
        self.fill_points()
        for point in tqdm(self.mapping):
            neighbors = self.get_neighbors(point)
            for neighbor in neighbors:
                p = self.index_mapping[point]
                n = neighbor
                self.graph.add_edge(p,n)
                if p < n:
                    self.edges.add((p,n))
                else:
                    self.edges.add((n,p))


class Convex_hull():
    def __init__(self,cycle):
        x_min = min([p[0] for p in cycle]) - 1
        x_max = max([p[0] for p in cycle]) + 1
        y_min = min([p[1] for p in cycle]) - 1 
        y_max = max([p[1] for p in cycle]) + 1
        z_min = min([p[2] for p in cycle]) - 1
        z_max = max([p[2] for p in cycle]) + 1
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

        self.Graph = simple_3D_Graph(x_min,x_max,y_min,y_max,z_min,z_max)
        self.graph_dense = nx.Graph()
        self.graph_convex_hull = nx.Graph()
        self.Graph.create_graph()
        self.cycle = [self.Graph.index_mapping[point] for point in cycle]
        self.create_dense_graph()
        self.create_initial_convex_hull()


    def next_f(self,f_0,new_f_0,edges):
        augment = False
        L = set()
        new_f = f_0.union(new_f_0)
        for p1 in tqdm(new_f):
            for p2 in f_0:
                if p1 != p2:
                    short_path = nx.shortest_path(self.Graph.graph, p1, p2)
                    for p in short_path:
                        if p not in f_0:
                            L.add(p)
                            augment = True
                    for i in range(len(short_path)-1):
                        v1 = short_path[i]
                        v2 = short_path[i+1]
                        if v1 < v2:
                            edges.add((v1,v2))
                        else:
                            edges.add((v2,v1))
        return edges,augment,f_0,L
    
    def create_dense_graph(self):
        f_0 = set(self.cycle)
        edges = set()
        for i in range(len(self.cycle)-1):
            e1 = self.cycle[i]
            e2 = self.cycle[(i+1)]
            if e1 < e2:
                edges.add((e1,e2))
            else:
                edges.add((e2,e1))
        new_f_0 = set()
        augment = True

        while augment:
            print(len(f_0))
            edges,augment,f_0,new_f_0 = self.next_f(f_0,new_f_0,edges)
            f_0 = f_0.union(new_f_0)
            print("next_f done")
        
        for e in edges:
            p1,p2 = e
            self.graph_dense.add_edge(p1,p2)

    def is_in_a_cube(self,node):
        edges = set()
        all_neighbours = set(self.graph_dense.neighbors(node))
        neighbours = list(self.graph_dense.neighbors(node))
        for n in neighbours:
            edges.add((node,n))
        for n1 in neighbours:
            for n2 in neighbours:
                if n1 != n2:
                    neighbours_n1 = set(self.graph_dense.neighbors(n1))
                    neighbours_n2 = set(self.graph_dense.neighbors(n2))
                    common_neighbours = neighbours_n1.intersection(neighbours_n2)
                    all_neighbours = all_neighbours.union(common_neighbours)
                    for n in common_neighbours:
                        edges.add((n1,n))
                        edges.add((n2,n))
        for n1 in all_neighbours:
            for n2 in all_neighbours:
                for n3 in all_neighbours:
                    if n1 != n2 and n1 != n3 and n2 != n3:
                        neighbours_n1 = set(self.graph_dense.neighbors(n1))
                        neighbours_n2 = set(self.graph_dense.neighbors(n2))
                        neighbours_n3 = set(self.graph_dense.neighbors(n3))
                        common_neighbours = neighbours_n1.intersection(neighbours_n2).intersection(neighbours_n3)
                        all_neighbours = all_neighbours.union(common_neighbours)
                        for n in common_neighbours:
                            edges.add((n1,n))
                            edges.add((n2,n))
                            edges.add((n3,n))
        return len(all_neighbours) == 27

    def is_in_a_square(self,node,slice_graph):
        all_neighbours = set(slice_graph.neighbors(node))
        neighbours = list(slice_graph.neighbors(node))
        for n1 in neighbours:
            for n2 in neighbours:
                if n1 != n2:
                    neighbours_n1 = set(slice_graph.neighbors(n1))
                    neighbours_n2 = set(slice_graph.neighbors(n2))
                    common_neighbours = neighbours_n1.intersection(neighbours_n2)
                    all_neighbours = all_neighbours.union(common_neighbours)
        return len(all_neighbours) == 9

    def convex_hull_2D(self,axe,pos):
        slice = [node for node in self.graph_dense.nodes() if self.Graph.mapping[node][axe] == pos]
        sub_graph_slice = self.graph_dense.subgraph(slice)
        convex_hull_slice_nodes = []
        convex_hull_slice = nx.Graph()
        for node in slice:
            if sub_graph_slice.degree(node) != 4:
                convex_hull_slice_nodes.append(node)
            elif not self.is_in_a_square(node,sub_graph_slice):
                convex_hull_slice_nodes.append(node)
        convex_hull_slice.add_nodes_from(convex_hull_slice_nodes)
        for e in sub_graph_slice.edges():
            u,v = e
            if u in convex_hull_slice_nodes and v in convex_hull_slice_nodes:
                convex_hull_slice.add_edge(u,v)
        return convex_hull_slice
    


    def create_initial_convex_hull(self):
        convex_hull_nodes = []

        for node in self.graph_dense.nodes():
            if self.graph_dense.degree(node) != 6:
                convex_hull_nodes.append(node)
            elif not self.is_in_a_cube(node):
                convex_hull_nodes.append(node)
                
        for e in self.graph_dense.edges():
            u,v = e
            if u in convex_hull_nodes and v in convex_hull_nodes:
                self.graph_convex_hull.add_edge(u,v)

    def get_level(self,convex_hull_slice,axe):
        levels = set([self.Graph.mapping[node][axe] for node in convex_hull_slice.nodes()])
        assert len(levels) == 1, "The convex hull is not a slice"
        return list(levels)[0]
        
    
    def change_level(self,convex_hull_slice,axe,level):
        """Change the level of the convex hull slice to level, Returns a new
        graph with the same edges and nodes but with the new level"""
        new_slice = nx.Graph()
        for node in convex_hull_slice.nodes():
            new_node = self.Graph.mapping[node]
            new_node = (new_node[0],new_node[1],level)
            new_slice.add_node(new_node)
        
        for edge in convex_hull_slice.edges():
            u,v = edge
            new_u,new_v = self.Graph.mapping[u],self.Graph.mapping[v]
            new_u = (new_u[0],new_u[1],level)
            new_v = (new_v[0],new_v[1],level)
            new_slice.add_edge(new_u,new_v)

        return new_slice


    def common_graph(self,convex_hull_1,convex_hull_2,axe):
        # Get the common projection
        axe1 = self.get_level(convex_hull_1,axe)
        axe2 = self.get_level(convex_hull_2,axe)
        assert axe1 != axe2, "The convex hulls are on the same level"
        assert axe1 == axe2 + 1 or axe1 == axe2 - 1, "The convex hulls are not on adjacent levels"
        convex_hull_1 = self.change_level(convex_hull_1,axe,axe2)
        convex_hull_2 = self.change_level(convex_hull_2,axe,axe2)
        common_graph = nx.Graph()

        for edge in convex_hull_1.edges():
            common_graph.add_edge(edge[0],edge[1])
        
        for edge in convex_hull_2.edges():
            common_graph.add_edge(edge[0],edge[1])

        return common_graph
    
    def set_edges_slice(self,convex_hull_slice_1,convex_hull_slice_2,axe):

        Set_edges = set()
        Border_edges = set()
        level_1 = self.get_level(convex_hull_slice_1,axe)
        level_2 = self.get_level(convex_hull_slice_2,axe)
        assert level_1 != level_2, "The slices are not on the same level"
        assert level_1 == level_2 + 1 or level_1 == level_2 - 1, "The slices are not on adjacent levels"
        min_level = min(level_1,level_2)
        max_level = max(level_1,level_2)
        for edge in self.graph_dense.edges():
            u,v = edge
            edge_axe = {self.Graph.mapping[u][axe],self.Graph.mapping[v][axe]}
            if edge_axe == {min_level,max_level}:
                Set_edges.add((u,v))
        All_nodes = set(convex_hull_slice_1.nodes()).union(set(convex_hull_slice_2.nodes()))
        
        for edge in tqdm(Set_edges):
            u,v = edge
            if u  in All_nodes or v in All_nodes:
                Border_edges.add(edge)
        
        return Border_edges

    def link_slices(self,convex_hull_slice_1,convex_hull_slice_2,axe):
        Border_edges = self.set_edges_slice(convex_hull_slice_1,convex_hull_slice_2,axe)
        level_1,level_2 = self.get_level(convex_hull_slice_1,axe),self.get_level(convex_hull_slice_2,axe)
        Inside_nodes = dict()
        Inside_nodes[level_1] = []
        Inside_nodes[level_2] = []
        Outside_nodes = dict()
        Outside_nodes[ level_1] = []
        Outside_nodes[level_2] = []
        All_nodes = set(convex_hull_slice_1.nodes()).union(set(convex_hull_slice_2.nodes()))
        for edge in tqdm(Border_edges):
            u,v = edge
            Inside_nodes[self.Graph.mapping[v][axe]].append(v)
            Inside_nodes[self.Graph.mapping[u][axe]].append(u)

        
        for node in All_nodes:
            Outside_nodes[self.Graph.mapping[node][axe]].append(node)
        
                
        return Inside_nodes,Outside_nodes
    
    def detect_tail(self,convex_hull_slice):
        deg_1 = [node for node in convex_hull_slice.nodes() if convex_hull_slice.degree(node) == 1]
        deg_1_edges = []
        while deg_1 != []:
            for n1 in deg_1 :
                deg_1_edges = deg_1_edges + list(convex_hull_slice.edges(n1))
                convex_hull_slice.remove_node(n1)
            deg_1 = [node for node in convex_hull_slice.nodes if convex_hull_slice.degree(node) == 1]
        return deg_1_edges


    def if_corner(self,convex_hull_slice,node):
        if convex_hull_slice.degree(node) != 2:
            return False
        n1,n2 = list(convex_hull_slice.neighbors(node))
        # Check if n1, n2, and node are on the same line
        x1, y1, z1 = self.Graph.mapping[n1]
        x2, y2, z2 = self.Graph.mapping[n2]
        x, y, z = self.Graph.mapping[node]
        if (x - x1) * (y2 - y1) == (y - y1) * (x2 - x1) and (x - x1) * (z2 - z1) == (z - z1) * (x2 - x1) and (y - y1) * (z2 - z1) == (z - z1) * (y2 - y1):
            return False
        else:
            return True
    
    
    def surface(self,convex_hull_slice,axe):
        surface = 0
        mid_edges = [(np.array(self.Graph.mapping[u]) + np.array(self.Graph.mapping[v]))/2 for u,v in convex_hull_slice.edges()]
        x,y = (axe + 1)%3,(axe + 2)%3
        x_min = min([self.Graph.mapping[node][x] for node in convex_hull_slice.nodes()])
        x_max = max([self.Graph.mapping[node][x] for node in convex_hull_slice.nodes()])
        x_begin = x_min + 0.5
        while x_begin < x_max:
            x_end = x_begin + 1
            y_min = min([edge[y] for edge in mid_edges if edge[x] == x_begin])
            y_max = max([edge[y] for edge in mid_edges if edge[x] == x_begin])
            surface = surface + (y_max - y_min)
            x_begin = x_end
        return surface



    def if_outside_corner(self,convex_hull_slice,node,axe):
        ind_n1,ind_n2 = list(convex_hull_slice.neighbors(node))
        n1,n2 = self.Graph.mapping[ind_n1],self.Graph.mapping[ind_n2]
        assert n1[axe] == n2[axe], "The neighbors are not on the same level"
        z = n1[axe]
        x,y = (axe + 1)%3,(axe + 2)%3
        new_x = [_x for _x in [n1[x],n2[x]] if _x != self.Graph.mapping[node][x]]
        new_y = [_y for _y in [n1[y],n2[y]] if _y != self.Graph.mapping[node][y]]
        assert len(new_x) == 1 and len(new_y) == 1, ("The neighbors are not on the same line",new_x,new_y,node)
        new_x,new_y = new_x[0],new_y[0]
        new_node = [0,0,0]
        new_node[x],new_node[y],new_node[axe] = new_x,new_y,z
        if tuple(new_node) not in self.Graph.mapping:
            print("new_node",new_node)
            self.Graph.mapping.append(tuple(new_node))
            self.Graph.index_mapping[tuple(new_node)] = len(self.Graph.mapping) - 1
        new_node = self.Graph.index_mapping[tuple(new_node)]

        # Check if changing node by new node reduces the surface
        surface1 = self.surface(convex_hull_slice,axe)
        new_convex_hull_slice = convex_hull_slice.copy()
        new_convex_hull_slice.remove_node(node)
        new_convex_hull_slice.add_node(new_node)
        new_convex_hull_slice.add_edge(ind_n1,new_node)
        new_convex_hull_slice.add_edge(ind_n2,new_node)
        surface2 = self.surface(new_convex_hull_slice,axe)

        if surface2 < surface1:
            return True
        else:
            return False
        
    def all_reducible_corners(self,convex_hull_slice,axe):
        tails = self.detect_tail(convex_hull_slice)
        corners = [node for node in convex_hull_slice.nodes() if self.if_corner(convex_hull_slice,node)]
        outside_corners = [node for node in corners if self.if_outside_corner(convex_hull_slice,node,axe)]
        for tail in tails:
            convex_hull_slice.add_edge(tail[0],tail[1])
        node_tails = set([node for node in convex_hull_slice.nodes() if convex_hull_slice.degree(node) == 1])
        return set(outside_corners).union(node_tails)

    


    def reduce_elastic_deg_2(self,convex_hull_elastic,convex_hull_slice,axe,node):
        ind_n1,ind_n2 = list(convex_hull_elastic.neighbors(node))
        n1,n2 = self.Graph.mapping[ind_n1],self.Graph.mapping[ind_n2]
        assert n1[axe] == n2[axe], "The neighbors are not on the same level"
        z = n1[axe]
        x,y = (axe + 1)%3,(axe + 2)%3
        new_x = [_x for _x in [n1[x],n2[x]] if _x != self.Graph.mapping[node][x]]
        new_y = [_y for _y in [n1[y],n2[y]] if _y != self.Graph.mapping[node][y]]
        assert len(new_x) == 1 and len(new_y) == 1, ("The neighbors are not on the same line",new_x,new_y,node)
        new_x,new_y = new_x[0],new_y[0]
        new_node = [0,0,0]
        new_node[x],new_node[y],new_node[axe] = new_x,new_y,z
        if tuple(new_node) not in self.Graph.mapping:
            print("new_node",new_node)
            self.Graph.mapping.append(tuple(new_node))
            self.Graph.index_mapping[tuple(new_node)] = len(self.Graph.mapping) - 1
        new_node = self.Graph.index_mapping[tuple(new_node)]
        convex_hull_elastic.add_node(new_node)
        convex_hull_elastic.add_edge(ind_n1,new_node)
        convex_hull_elastic.add_edge(ind_n2,new_node)
        convex_hull_elastic.remove_node(node)

        convex_hull_slice.add_node(new_node)
        convex_hull_slice.add_edge(ind_n1,new_node)
        convex_hull_slice.add_edge(ind_n2,new_node)
    
    def reduce_elastic_deg_1(self,convex_hull_elastic,convex_hull_slice,axe,node):
        convex_hull_elastic.remove_node(node)
        
    

        
        
        


    def elastic_band(self,inside_nodes,convex_hull_slice,axe):

        elastic_band = convex_hull_slice.copy()
        inside_nodes = set(inside_nodes)
        reducible_nodes = [node for node in self.all_reducible_corners(elastic_band,axe) if node not in inside_nodes and elastic_band.degree(node) < 3]

        while reducible_nodes:
            

            node_to_remove = reducible_nodes.pop()

            if  elastic_band.degree(node_to_remove) == 2:
                self.reduce_elastic_deg_2(elastic_band,convex_hull_slice,axe,node_to_remove)

            elif elastic_band.degree(node_to_remove) == 1:
                self.reduce_elastic_deg_1(elastic_band,convex_hull_slice,axe,node_to_remove)

            reducible_nodes = [node for node in self.all_reducible_corners(elastic_band,axe) if node not in inside_nodes and elastic_band.degree(node) < 3]
        

            
    def convex_hull_3D_full(self):
        axe = 2
        Convex = dict()
        Edges = set()
        for z in range(self.z_min+1,self.z_max-1):
            level1 = z
            level2 = z+1
            convex_hull_slice_1 = self.convex_hull_2D(axe,level1)
            convex_hull_slice_2 = self.convex_hull_2D(axe,level2)
            convex_hull_slice_1 = clean_slice(convex_hull_slice_1)
            convex_hull_slice_2 = clean_slice(convex_hull_slice_2)
            inside_nodes,_ = self.link_slices(convex_hull_slice_1,convex_hull_slice_2,axe)
            inside_nodes_1 = inside_nodes[level1]
            inside_nodes_2 = inside_nodes[level2]

            self.elastic_band(inside_nodes_1,convex_hull_slice_1,axe)
            self.elastic_band(inside_nodes_2,convex_hull_slice_2,axe)
            
            Convex[level1] = convex_hull_slice_1
            Convex[level2] = convex_hull_slice_2

            Border_edges = self.set_edges_slice(convex_hull_slice_1,convex_hull_slice_2,axe)
            Edges = Edges.union(Border_edges) 
        # Fill the first and last layers
        self.elastic_band({},Convex[self.z_min+1],axe)
        self.elastic_band({},Convex[self.z_max-1],axe)
        
        G_full_convex_hull_graph = nx.Graph()
        for edge in Edges:
            u,v = edge
            G_full_convex_hull_graph.add_edge(u,v)

        for z in range(self.z_min+1,self.z_max):
            convex_hull_slice = Convex[z]
            for edge in convex_hull_slice.edges():
                u,v = edge
                G_full_convex_hull_graph.add_edge(u,v)
        
        return G_full_convex_hull_graph
        
           


            
    
                
     
    
    def convex_set_edges(self,set_edges,axe):
        levels = []
        for edge in set_edges:
            u,v = edge
            levels.append(self.Graph.mapping[u][axe])
            levels.append(self.Graph.mapping[v][axe])
        assert len(set(levels)) == 2, "The set of edges is not a slice"
        
    def in_same_square(self,e1,e2):
        u1,v1 = e1
        u2,v2 = e2
        squre_graph = self.graph_dense.subgraph([u1,v1,u2,v2])
        cycles = nx.cycle_basis(squre_graph)
        if len(cycles) == 1:
            return True
        elif len(set([u1,v1,u2,v2])) == 3:
            set_x = set()
            set_y = set()
            set_z = set()
            for point in [u1,v1,u2,v2]:
                x,y,z = self.Graph.mapping[point]
                set_x.add(x)
                set_y.add(y)
                set_z.add(z)
            if len(set_x) == 1 and len(set_y) == 1:
                return False
            if len(set_x) == 1 and len(set_z) == 1:
                return False
            if len(set_y) == 1 and len(set_z) == 1:
                return False
            return True
        return False



    def new_graph(self,new_cut):
        Couple_edges = []
        for e1 in new_cut:
            for e2 in new_cut:
                if e1 != e2:
                    if self.in_same_square(e1,e2):
                        Couple_edges.append((e1,e2))

        graph_edges = nx.Graph()

        for e1,e2 in Couple_edges:
            n1 = (np.array(self.Graph.mapping[e1[0]]) + np.array(self.Graph.mapping[e1[1]]))/2
            n2 = (np.array(self.Graph.mapping[e2[0]]) + np.array(self.Graph.mapping[e2[1]]))/2
            graph_edges.add_edge(tuple(n1),tuple(n2))
        return graph_edges
        



## End of the class Convex_hull
                
def in_same_square(convex_hull_slice,e1,e2):
    u1,v1 = e1
    u2,v2 = e2
    squre_graph = convex_hull_slice.subgraph([u1,v1,u2,v2])
    return nx.is_connected(squre_graph)

def list_in_same_square(convex_hull_slice,edges,e):
    return [edge for edge in edges if _in_same_square(convex_hull_slice,edge,e)]
 
def verify(convex_hull_slice):
    verified = nx.is_connected(convex_hull_slice)
    for node in convex_hull_slice.nodes():
        if convex_hull_slice.degree(node) != 2:
            verified = False
    return verified

def clean_slice(convex_hull_slice):
        
    # Get rid of the 1 degree paths 
    Nodes = set(convex_hull_slice.nodes())
    deg_1 = [node for node in Nodes if convex_hull_slice.degree(node) == 1]
    deg_1_edges = []
    while deg_1 != []:
        for n1 in deg_1 :
            deg_1_edges = deg_1_edges + list(convex_hull_slice.edges(n1))
            convex_hull_slice.remove_node(n1)
        deg_1 = [node for node in convex_hull_slice.nodes if convex_hull_slice.degree(node) == 1]
    new_graph = nx.Graph()
    if convex_hull_slice.number_of_nodes() == 0:
        print("Convex hull is empty")
        
    
    # Get rid of the 2 degree paths
    else:
        cycle = sorted(list(nx.simple_cycles(convex_hull_slice)), key = lambda s: len(s)) 
        Edges = []
        if len(cycle) == 0:
            print("No cycle found")
        longest_cycle = cycle[-1]
        
        for i in range(len(longest_cycle)):
            Edges.append((longest_cycle[i],longest_cycle[(i+1)%len(longest_cycle)]))
        new_graph.add_edges_from(Edges)
    
    
    # Restore 1 degree paths
    for e in deg_1_edges:
        u,v = e
        new_graph.add_edge(u,v)
    
    return new_graph

def color(graph):
    convex_hull_nodes = graph.nodes()
    visited = set(convex_hull_nodes)
    v1,v2 = np.random.choice(convex_hull_nodes,size=2)
    visited.remove(v1)
    visited.remove(v2)
    layer_1 = set([v1])
    layer_2 = set([v2])
    color_1 = set([v1])
    color_2 = set([v2])
    while visited:
        new_layer_1 = set()
        new_layer_2 = set()
        for v in layer_1:
            for u in graph.neighbors(v):
                if u in visited:
                    visited.remove(u)
                    new_layer_1.add(u)
    
        for v in layer_2:
            for u in graph.neighbors(v):
                if u in visited:
                    visited.remove(u)
                    new_layer_2.add(u)

        layer_1 = new_layer_1
        layer_2 = new_layer_2

        color_1 = color_1.union(layer_1)
        color_2 = color_2.union(layer_2)
    
    return color_1,color_2

def cut(out_graph,dense_edges,cap):
    color1,color2 = color(out_graph)
    G_flot = nx.Graph()
    s = -1
    t = -2

    for p in color1:
        G_flot.add_edge(s,p,capacity=cap)
        
    for p in color2:    
        G_flot.add_edge(p,t,capacity = cap)

    for e in dense_edges:
        u,v = e
        G_flot.add_edge(u,v,capacity=1)



    cut_value, (set_S, set_T) = nx.minimum_cut(G_flot, s, t)

    # Find the edges in the minimum cut
    min_cut_edges = [(u, v) for u, v in G_flot.edges() if (u in set_S and v in set_T) or (v in set_S and u in set_T)]
    return min_cut_edges

def in_same_square(self,e1,e2):
    u1,v1 = e1
    u2,v2 = e2
    squre_graph = self.graph_dense.subgraph([u1,v1,u2,v2])
    cycles = nx.cycle_basis(squre_graph)
    if len(cycles) == 1:
        return True
    elif len(set([u1,v1,u2,v2])) == 3:
        set_x = set()
        set_y = set()
        set_z = set()
        for point in [u1,v1,u2,v2]:
            x,y,z = self.Graph.mapping[point]
            set_x.add(x)
            set_y.add(y)
            set_z.add(z)
        if len(set_x) == 1 and len(set_y) == 1:
            return False
        if len(set_x) == 1 and len(set_z) == 1:
            return False
        if len(set_y) == 1 and len(set_z) == 1:
            return False
        return True
    return False


