from networkx import Graph
from ...Plateau_Problem.Triangulation_Meshing.PointList import *
from .CreatingCycles import *
from tqdm import tqdm
from ...utils import *
from ..._3DLoops._3dpu_using_dfs import *
from ...Block_edges.block_edges import *
import gurobipy as gp
from gurobipy import Model, GRB, quicksum

def points(cycle,C):
    """Function for the test of the MIP on a cycle of the object C"""
    points = PointList()
    for res in cycle:
        points.add_point(transform_res_to_point(C.mapping[res]))
    return points

def fill_cycle(cycle,C):
    """Function for filling up the cycle using the construction of the Graph cycles C"""
    points = PointList()
    for res in cycle:
        points.add_point(transform_res_to_point(C.mapping[res]))
    M = Edge_Flipping(points,len(points.points))
    M.create_quadrilaterals()
    M.split_quadrilateral()
    M.canonic_representation_from_mesh()
    M.clean_triangles()
    M.fill_edges()
    M.update_weights()
    M.mapping = [np.array(o) for o in M.mapping]
    E = Block_edges(M.triangles,M.mapping)
    E.block_all_the_edges()
    Edges = (E.blocked_edges)
    return Edges,M


def minimize_edges_MIP(Edges, num_threads=1):
    Marked_edges = []
    Blocked_edges = Edges
    G = GraphGrid3D(Blocked_edges, [])

    # Create a new model
    model = Model("Graph Problem")

    # Suppress Gurobi solver output
    model.setParam('OutputFlag', 1)
    model.setParam('Threads', num_threads)
    model.setParam('LazyConstraints', 1)

    # Define the decision variables
    x = model.addVars(G.edges, vtype=GRB.BINARY, name="x")

    # Define the objective function
    model.setObjective(quicksum(x[i] for i in G.edges), GRB.MINIMIZE)

    # Initialize Graph_Cycles object
    GC = Graph_Cycles(Blocked_edges, Marked_edges)
    cpt = [0]

    # Callback function to add lazy constraints
    def lazy_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            Marked_edges = []
            for edge in G.edges:
                if model.cbGetSolution(x[edge]) > 0.5:
                    Marked_edges.append([G.mapping[edge[0]], G.mapping[edge[1]]])

            GC = Graph_Cycles(Blocked_edges, Marked_edges)

            for cycle in GC.b_1:
                cpt[0] += 1
                cycle_edges = [(min(cycle[i], cycle[i + 1]), max(cycle[i], cycle[i + 1])) for i in range(len(cycle) - 1)]
                model.cbLazy(quicksum(x[edge] for edge in cycle_edges) >= 1)

    initial_cycle = min(GC.b_1, key=len)
    initial_cycle_edges = [(min(initial_cycle[i], initial_cycle[i + 1]), max(initial_cycle[i], initial_cycle[i + 1])) for i in range(len(initial_cycle) - 1)]
    model.addConstr(quicksum(x[edge] for edge in initial_cycle_edges) >= 1)

    # Optimize the model with the lazy constraints
    model.optimize(lazy_callback)

    if model.status != GRB.OPTIMAL:
        print("No optimal solution found.")

    # Update Marked_edges based on the final solution
    Marked_edges = []
    for edge in G.edges:
        if x[edge].X > 0.5:
            Marked_edges.append([G.mapping[edge[0]], G.mapping[edge[1]]])

    return Marked_edges
        

