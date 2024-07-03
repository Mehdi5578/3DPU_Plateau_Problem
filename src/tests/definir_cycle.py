from .ConstructingSurfaces import *
from .CreatingCycles import *
from _3DLoops._3dpu_using_dfs import *
import nibabel as nb

ROOT = "../"



def transform_res_to_point(res):
    x,y,z,ax,_ = res
    if ax == 0:
        return (x,y+0.5,z+0.5)
    elif ax == 1:
        return (x+0.5,y,z+0.5)
    else:
        return (x+0.5,y+0.5,z)

class Refine_cycle:
    def __init__(self, cyc):
        self.squares = self.transform_to_squares(cyc)
        self.center_cycle = []
        for square in self.squares:
            a1,a2,a3,a4,_ = square
            x = (a1[0] + a2[0] + a3[0] + a4[0])/4
            y = (a1[1] + a2[1] + a3[1] + a4[1])/4
            z = (a1[2] + a2[2] + a3[2] + a4[2])/4
            self.center_cycle.append((x,y,z))
        nodes = self.get_nodes_from_square(self.squares)
        self.x_min = min([x for x,y,z in nodes])
        self.x_max = max([x for x,y,z in nodes])
        self.y_min = min([y for x,y,z in nodes])
        self.y_max = max([y for x,y,z in nodes])
        self.z_min = min([z for x,y,z in nodes])
        self.z_max = max([z for x,y,z in nodes])

        self.grid_graph = simple_3D_Graph(self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    def transform_to_squares(self,cycle):
        squares = []
        for i in range(len(cycle)-1):
            x,y,z,ax,_ = cycle[i]	
            square = [(x,y,z)]
            if ax == 0:
                square.append((x,y+1,z))
                square.append((x,y+1,z+1))
                square.append((x,y,z+1))
            elif ax == 1:
                square.append((x,y,z+1))
                square.append((x+1,y,z+1))
                square.append((x+1,y,z))
            else:
                square.append((x+1,y,z))
                square.append((x+1,y+1,z))
                square.append((x,y+1,z))
            square.append(square[0])
            squares.append(square)
        return squares
    

    



    def get_nodes_from_square(self,squares):
        nodes = []
        for square in squares:
            for point in square:
                nodes.append(point)
        return nodes

    def edges_from_squares(self):
        edges = []
        for square in self.squares:
            for i in range(4):
                edges.append((square[i], square[i+1]))
        return edges

    
    def in_same_square_here(self,e1,e2):
        u1,v1 = e1
        u2,v2 = e2
        u1= self.grid_graph.index_mapping[u1]
        v1= self.grid_graph.index_mapping[v1]
        u2= self.grid_graph.index_mapping[u2]
        v2= self.grid_graph.index_mapping[v2]
        squre_graph = self.grid_graph.graph.subgraph([u1,v1,u2,v2])
        cycles = nx.cycle_basis(squre_graph)
        if len(cycles) == 1:
            return True
        elif len(set([u1,v1,u2,v2])) == 3:
            set_x = set()
            set_y = set()
            set_z = set()
            for point in [u1,v1,u2,v2]:

                x,y,z = self.grid_graph.mapping[point]
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
    
    def new_graph(self,set_edges):
        
        Couple_edges = []
        for e1 in set_edges:
            for e2 in set_edges:
                if e1 != e2:
                    if self.in_same_square_here(e1,e2):
                        Couple_edges.append((e1,e2))

        graph_edges = nx.Graph()

        for e1,e2 in Couple_edges:
            n1 = (np.array(e1[0]) + np.array(e1[1]))/2
            n2 = (np.array(e2[0]) + np.array(e2[1]))/2
            
            graph_edges.add_edge(tuple(n1),tuple(n2))
        return graph_edges



class Clean_Cycles:

    def __init__(self,data):
        self.data = data
        self.Mapping_index_data = dict()

        for i in range(len(data.mapping)):
            self.Mapping_index_data[data.mapping[i]] = i
        
        self.data.create_graph()
    
    def graph_of_residuals(self,cycle):
        """
        Gives a directed subgraph of the inital Graph Res_graph in data
        """
        new_graph = nx.DiGraph()
        for point in cycle:
            Edges_point = self.data.Res_graph[point]
            for edge in Edges_point:
                if edge in cycle:
                    new_graph.add_edge(point,edge)
        return new_graph

    
    def detangle_new_graph(self,cycle):
        """
        This function takes a graph and returns the edges that point to more than two edges
        """
        new_graph = self.graph_of_residuals(cycle)
        Edges_to_remove = []
        Edges_of_cycle = [set(cycle[i:i+2]) for i in range(len(cycle)-1)]
        for point in new_graph.nodes:
            if new_graph.out_degree(point) == 2:
                print(point)
                p1,p2 = list(new_graph.successors(point))
                if {point,p1} in Edges_of_cycle:
                    Edges_to_remove.append((point,p1))
                else:
                    assert {point,p2} in Edges_of_cycle
                    Edges_to_remove.append((point,p2))
        return Edges_to_remove
    





    
    


    