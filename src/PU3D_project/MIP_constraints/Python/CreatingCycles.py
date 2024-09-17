import networkx as nx
from .CreatingGraph import *
from tqdm import tqdm
from collections import defaultdict, deque


class Graph_Cycles:
    def __init__(self, Edges, Marked_edges):
        self.cycles = []
        self.cycles_index = {}
        self.dict_blocked_edges = {}
        self.Graph_cycles = nx.Graph()
        self.edge_in_cycle = {}

        G = GraphGrid3D(Edges,Marked_edges)
        self.mapping_GC = G.mapping
        self.edges = G.edges
        self.Graph = G.Graph
        self.marked_edges = Marked_edges

        self.blocked_edges = set(G.blocked_edges)
        new_G = nx.Graph()
        self.b_1 = set()
        self.b_2 = set()
        new_G.add_edges_from(list(G.edges))

        cycles_base = nx.cycle_basis(new_G)
        
        cycles_base = [tuple(cl+[cl[0]]) for cl in cycles_base]
        self.cycles = cycles_base

        for cycle in self.cycles:
            parite = self.f(cycle)
            if parite == 1:
                self.b_1.add(cycle)
            elif parite == 0:
                self.b_2.add(cycle)
    
    
    def spanning_tree_BFS(self):
        dict_nodes = {}
        dict_passage = {}
        visited = set()
        edges = set()

        # Initialize dict_nodes and dict_passage for all nodes
        for node in range(len(self.mapping_GC)):
            dict_nodes[node] = 0
            dict_passage[node] = None

        start_node = np.random.randint(0, len(self.mapping_GC))
        deque_nodes = deque([start_node])
        visited.add(start_node)

        while deque_nodes:
            node = deque_nodes.popleft()
            for neighbor in self.Graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    deque_nodes.append(neighbor)
                    edge = (min(node, neighbor), max(node, neighbor))
                    edges.add(edge)
                    blocked = 1 if edge in self.blocked_edges else 0
                    dict_nodes[neighbor] = dict_nodes[node] + blocked
                    dict_passage[neighbor] = node

        return dict_nodes, dict_passage, edges

    def detect_odd_blocked_cycles(self):
        # First, build the spanning tree using BFS
        dict_nodes, dict_passage, edges_in_tree = self.spanning_tree_BFS()

        parity = {}  # Parity of blocked edges from root to each node
        for node in dict_nodes:
            parity[node] = dict_nodes[node] % 2  # 0 for even, 1 for odd

        cycles_with_odd_blocked_edges = []

        # Process all edges to find non-tree edges (back edges)
        for node in range(len(self.mapping_GC)):
            for neighbor in self.Graph[node]:
                if neighbor != dict_passage.get(node, None):  # Avoid parent edge
                    edge = (min(node, neighbor), max(node, neighbor))
                    if edge not in edges_in_tree:
                        # This is a non-tree edge, forms a fundamental cycle
                        u, v = node, neighbor

                        # Calculate total parity in the cycle
                        blocked_edge = 1 if edge in self.blocked_edges else 0
                        total_parity = parity[u] ^ parity[v] ^ blocked_edge

                        if total_parity == 1:
                            # Cycle has an odd number of blocked edges
                            cycle_nodes = self.get_cycle_nodes(u, v, dict_passage)
                            cycles_with_odd_blocked_edges.append(cycle_nodes)

        return cycles_with_odd_blocked_edges

    def get_cycle_nodes(self, u, v, dict_passage):
        # Function to reconstruct the cycle from node u to node v
        path_u = []
        path_v = []

        # Traverse from u to root, recording the path
        current = u
        while current is not None:
            path_u.append(current)
            current = dict_passage[current]

        # Traverse from v to root, recording the path
        current = v
        while current is not None:
            path_v.append(current)
            current = dict_passage[current]

        # Find the lowest common ancestor (LCA)
        set_u = set(path_u)
        lca = None
        for node in path_v:
            if node in set_u:
                lca = node
                break

        # Build the cycle path from u to v via LCA
        cycle_path = []
        for node in path_u:
            cycle_path.append(node)
            if node == lca:
                break
        cycle_path.reverse()  # Reverse to get path from LCA to u

        # Append path from LCA to v
        index = path_v.index(lca)
        cycle_path.extend(path_v[:index])

        return cycle_path
    


    
    
    # def b_1(self):
    #     dict_nodes,dict_passage,edges = self.spanning_tree_BFS()
    #     remaining_edges = self.edges - edges
    #     cycle_basis_1 = []
    #     pass
        


        


        
        
        


        

        
    def f(self,cycle):
        cpt  = 0
        for i in range(len(cycle)):
            edge = (min(cycle[i],cycle[(i+1)%len(cycle)]),max(cycle[i],cycle[(i+1)%len(cycle)]))
            cpt = cpt + (edge in self.blocked_edges)
        return cpt % 2





            
        
        