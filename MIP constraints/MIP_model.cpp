#include "MIP_model.hpp"
#include "CreatingGraph.hpp"
#include "CreatingGraph.cpp"

#include "gurobi_c++.h"
using namespace std;




int
main(int   argc,
     char *argv[])

{
  MIP_model::Graph g(208,208,96);
  g.CreateGraph(208,208,96);
  int number_edges  = g.countEdges();
  int number_vertices = g.adjList.size();
  cout << "Number of vertices: " << number_vertices << endl;
  cout << "Number of edges: " << number_edges << endl;
  try {

    // Create an environment
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "mip1.log");
    env.start();

    // Create an empty model
    GRBModel model = GRBModel(env);
    int size = 12376832;
    // Create 3D variable
    GRBVar x[size];


    for (int i = 0; i < size; i++){
        x[i] = model.addVar(0.0, 100,0,GRB_INTEGER, "x_" + to_string(i) );
      }
  

    model.update();
    // GRBVar y = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "y");
    // GRBVar z = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "z");

    // Set objective: maximize x.y + 2 y.z
    GRBQuadExpr Obj = 0;
    for (int i = 0; i < size; i++){
        Obj += x[i]* x[i];
      
    }
    model.setObjective(Obj, GRB_MAXIMIZE);

    // Add constraint: x + 2 y + 3 z <= 4
    model.addConstr(x[0]- 2 * x[1] + 3 * x[143] <= 1, "c0");

    for (int i = 0 ; i < size; i++){
          model.addConstr(x[i]  + x[i+124] <= 1, "c1");
      }

    // Add constraint: x + y >= 1

    // Optimize model
    model.optimize();

    for (int i = 0; i < size; i++){
        if (x[i].get(GRB_DoubleAttr_X) == 1){
          cout << "x_" + to_string(i)  << " = " << x[i].get(GRB_DoubleAttr_X) << endl;
         
      }
    }

    cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

  } catch(GRBException e) {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch(...) {
    cout << "Exception during optimization" << endl;
  }

  return 0;
}



// int main() {
//     // Create a graph given in the above diagram
//     auto start = std::chrono::high_resolution_clock::now();
//     MIP_model::Graph g(208,208,96);
//     g.CreateGraph(208,208,96);
//     int number_edges  = g.countEdges();
//     auto stop = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = stop - start;
//     std::cout << "Time to create the graph: " << elapsed.count() << " s\n";
//     int cpt_3 = 0;
//     int cpt_4 = 0;
//     int cpt_5 = 0;
//     int cpt_6 = 0;
//     int cpt_total = 0;
//     for (int i=0; i<g.adjList.size(); i++){
//         if (g.adjList[i].size()==3){
//             cpt_3++;
//         }
//         if (g.adjList[i].size()==4){
//             cpt_4++;
//         }
//         if (g.adjList[i].size()==5){
//             cpt_5++;
//         }
//         if (g.adjList[i].size()==6){
//             cpt_6++;
//         }
//         cpt_total += 1;
//     }
//     std::cout << "Number of vertices with 3 neighbors: " << cpt_3 << std::endl;
//     std::cout << "Number of vertices with 4 neighbors: " << cpt_4 << std::endl;
//     std::cout << "Number of vertices with 5 neighbors: " << cpt_5 << std::endl;
//     std::cout << "Number of vertices with 6 neighbors: " << cpt_6 << std::endl;
//     std::cout << "Total number of vertices: " << cpt_total << std::endl;
//     std::cout << "Total number of edges: " << number_edges << std::endl;
//     return 0;
// }
