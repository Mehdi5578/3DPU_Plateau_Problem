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
        self.edges = G.edges
        self.Graph = G.Graph
        self.marked_edges = Marked_edges

        self.blocked_edges = set(G.blocked_edges)
        self.b_1 = set()
        self.b_2 = set()

        self.Euler = []
        self.depth = []
        self.first_index = {}
        self.sparse_table = []
        self.log_table = [] 


        self.b_1 = set()
        self.find_b1_abrupt()


    def find_b1_abrupt(self):
        new_G = nx.Graph()
        new_G.add_edges_from(list(self.edges))
        cycles_base = nx.cycle_basis(new_G)
        cycles_base = [tuple(cl+[cl[0]]) for cl in cycles_base]
        self.cycles = cycles_base
        for cycle in self.cycles:
            parite = self.f(cycle)
            if parite == 1:
                self.b_1.add(cycle)
            elif parite == 0:
                self.b_2.add(cycle)
    
    def f(self,cycle):
        cpt = 0
        for i in range(len(cycle)-1):
            if (cycle[i],cycle[i+1]) in self.blocked_edges or (cycle[i+1],cycle[i]) in self.blocked_edges:
                cpt += 1
        return cpt % 2
    
    def create_tree(self):
        u = np.random.randint(len(self.Graph))
        T_i = dict()
        Parent_i = dict()
        dict_i = {}
        dict_i[u] = 0
        queue = deque([u])
        visited = set()
        while queue :
            v = queue.popleft()
            
            for w in self.Graph[v]:
                if w not in visited:
                    visited.add(w)
                    if v in T_i:
                        T_i[v].append(w)
                    else:
                        T_i[v] = [w]
                    if w in Parent_i:
                        Parent_i[w].append(v)
                    else:
                        Parent_i[w] = [v]
                    queue.append(w)

                    if (v,w) in self.blocked_edges or (w,v) in self.blocked_edges:
                        dict_i[w] = dict_i[v] + 1
                    else:
                        dict_i[w] = dict_i[v]
        return T_i,Parent_i,dict_i
    
    def edges_tree(self):
        edges = set()
        for u in self.tree:
            for v in self.tree[u]:
                edges.add((min(u,v),max(u,v)))
        return edges
    
    def dfs_euler(self, u, depth, tree, visited):
        self.Euler.append(u)
        self.depth.append(depth)
        if u not in self.first_index:
            self.first_index[u] = len(self.Euler) - 1
        visited.add(u)

        for v in self.Graph[u]:
            if (v in tree and u in tree[v]) or (u in tree and v in tree[u]):
                if v not in visited:
                    self.dfs_euler(v, depth + 1, tree, visited)
                    self.Euler.append(u)
                    self.depth.append(depth)

                    
    def build_sparse_table(self):
        n = len(self.depth)
        self.log_table = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log_table[i] = self.log_table[i // 2] + 1

        log_n = self.log_table[n] + 1
        self.sparse_table = [[0] * log_n for _ in range(n)]

        for i in range(n):
            self.sparse_table[i][0] = i

        j = 1
        while (1 << j) <= n:
            i = 0
            while i + (1 << j) - 1 < n:
                left = self.sparse_table[i][j - 1]
                right = self.sparse_table[i + (1 << (j - 1))][j - 1]
                if self.depth[left] < self.depth[right]:
                    self.sparse_table[i][j] = left
                else:
                    self.sparse_table[i][j] = right
                i += 1
            j += 1
    
    def query_rmq(self, l, r):
        j = self.log_table[r - l + 1]
        left = self.sparse_table[l][j]
        right = self.sparse_table[r - (1 << j) + 1][j]
        if self.depth[left] < self.depth[right]:
            return left
        else:
            return right

    def query_lca(self, u, v):
        l = min(self.first_index[u], self.first_index[v])
        r = max(self.first_index[u], self.first_index[v])
        idx = self.query_rmq(l, r)
        return self.Euler[idx] 


    def build_path(self,u,v):
        lca = self.query_lca(u,v)
        path_u = [u]
        path_v = [v]

        while u != lca:
            path_u.append(self.parent[u][0])
            u = self.parent[u][0]
        
        while v != lca:
            path_v.append(self.parent[v][0])
            v = self.parent[v][0]
        
        path_v.reverse()
        return path_u + path_v

    def build_b_1(self):
        for edge in tqdm(self.edges):
            i,j = edge
            edge = (min(i,j),max(i,j))
            if edge not in self.tree_edges:
                u,v = edge
                Nuv = self.dict_i[u] + self.dict_i[v] 
                if (u,v) in self.blocked_edges or (v,u) in self.blocked_edges:
                    Nuv += 1
                
                if Nuv % 2 == 1:
                    path = self.build_path(u,v)
                    self.b_1.add(tuple(path))
        return self.b_1

    def find_b1(self):
        self.tree,self.parent,self.dict_i = self.create_tree()
        self.tree_edges = self.edges_tree()
        self.dfs_euler(0,0,self.tree,set())
        self.build_sparse_table()
        self.b_1 = self.build_b_1()
        return self.b_1








            
        
        