#include "MIP_model.hpp"
#include "CreatingGraph.hpp"
#include "CreatingGraph.cpp"


//this is outerclass class called MIP_model

int main() {
    // Create a graph given in the above diagram
    auto start = std::chrono::high_resolution_clock::now();
    MIP_model::Graph g(3,3,1);
    g.CreateGraph(3,3,1);
    int number_edges  = g.countEdges();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    std::cout << "Time to create the graph: " << elapsed.count() << " s\n";
    int cpt_3 = 0;
    int cpt_4 = 0;
    int cpt_5 = 0;
    int cpt_6 = 0;
    int cpt_total = 0;
    for (int i=0; i<g.adjList.size(); i++){
        if (g.adjList[i].size()==3){
            cpt_3++;
        }
        if (g.adjList[i].size()==4){
            cpt_4++;
        }
        if (g.adjList[i].size()==5){
            cpt_5++;
        }
        if (g.adjList[i].size()==6){
            cpt_6++;
        }
        cpt_total += 1;
    }
    std::cout << "Number of vertices with 3 neighbors: " << cpt_3 << std::endl;
    std::cout << "Number of vertices with 4 neighbors: " << cpt_4 << std::endl;
    std::cout << "Number of vertices with 5 neighbors: " << cpt_5 << std::endl;
    std::cout << "Number of vertices with 6 neighbors: " << cpt_6 << std::endl;
    std::cout << "Total number of vertices: " << cpt_total << std::endl;
    std::cout << "Total number of edges: " << number_edges << std::endl;
    

    
}
