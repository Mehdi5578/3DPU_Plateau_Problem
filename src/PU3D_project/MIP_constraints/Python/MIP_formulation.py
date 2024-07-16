import sys
from ...Plateau_Problem.Triangulation_Meshing.PointList import *
from .CreatingCycles import *
from tqdm import tqdm
from ...utils import *
from ..._3DLoops._3dpu_using_dfs import *
from ...Block_edges.block_edges import *
from .CreatingCycles import *
import gurobipy as gp
from gurobipy import Model, GRB, quicksum


def points(cycle,C):
    points = PointList()
    for res in cycle:
        points.add_point(transform_res_to_point(C.mapping[res]))
    return points

def fill_cycle(cycle,C):
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

def minimize_edges_MIP(Edges,time_limit,num_threads=1):
    Marked_edges = []
    Blocked_edges = Edges
    GC = Graph_Cycles(Blocked_edges, Marked_edges)
    G = GraphGrid3D(Blocked_edges, [])

    # Create a new model
    model = Model("Graph Problem")

    # Suppress Gurobi solver output
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', num_threads)             
    # Match this to the --cpus-per-task value in SLURM script
    # model.setParam('LazyConstraints', 1) 

    # Define the decision variables
    x = model.addVars(G.edges, vtype=GRB.BINARY, name="x")

    # Define the objective function
    model.setObjective(quicksum(x[i] for i in G.edges), GRB.MINIMIZE)

    # Function to add lazy constraints
    def add_cycle_constraints(model, GC):
        cpt = 0 
        while GC.b_1:

            # Get the smallest cycle in GC.b_1
            added_cycles = [min(GC.b_1, key=len)]
            edges_in_cycle = []
            for cycle in added_cycles:
                for i in range(len(cycle) - 1):
                    node, next_node = cycle[i], cycle[i + 1]
                    edge = (min(node, next_node), max(node, next_node))
                    edges_in_cycle.append(edge)
            
            # Add a lazy constraint to ensure at least one edge in the cycle is selected
            model.addConstr(gp.quicksum(x[edge] for edge in edges_in_cycle) >= 1)
            
            # Solve the model again
            model.optimize()
            
            # Check the new solution
            if model.status == GRB.OPTIMAL:

                # Update Marked_edges based on the current solution
                new_marked = []
                for edge in G.edges:
                    if x[edge].X > 0.5:
                        new_marked.append([G.mapping[edge[0]], G.mapping[edge[1]]])
                
                # Update the graph cycles with the new marked edges
                GC = Graph_Cycles(Blocked_edges, new_marked)
                # if cpt % 5 == 0:
                #     print(len(new_marked),len(GC.b_1))
                # cpt += 1
            else:
                print("No optimal solution found.")

    # Initialize Graph_Cycles object
    GC = Graph_Cycles(Blocked_edges, Marked_edges)

    # Optimize the model without lazy constraints first
    model.optimize()

    # Add cycle constraints until GC.b_1 is empty
    add_cycle_constraints(model, GC)

    # Print the final solution
    if model.status == GRB.OPTIMAL:
        # print("Optimal solution found:")
        for edge in G.edges:
            if x[edge].X > 0.5:
                Marked_edges.append([G.mapping[edge[0]], G.mapping[edge[1]]])
    else:
        print("Time limit exceeded we keep old Marked_edges")
        Marked_edges = Blocked_edges
    
    return Marked_edges




        

