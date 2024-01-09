#include "MIP_model.hpp"
#include <iostream>
#include <tuple>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <unordered_set>
#include <chrono>




// Constructor: initializes the number of vertices and resizes the adjacency list
Graph::Graph(int N, int M, int L) : adjList(N * M * L) {

}


// from N*M*L create a mapping of the 3D points
// void Graph::CreateMapping(int N,int M, int L){
//     for (int i=0; i<N; i++){
//         for (int j=0; j<M; j++){
//             for (int k=0; k<L; k++){
//                 int key = i*M*L + j*L + k;
//                 mapping[key] = key;
                
//             }
//         }
//     }        
// }

std::tuple<int,int,int> Graph::getCoordinates(int index, int N, int M, int L){
    int i = index/(M*L);
    int j = (index-i*M*L)/L;
    int k = index-i*M*L-j*L;
    return std::make_tuple(i,j,k);
}


int Graph::getIndex(int i, int j, int k, int N, int M, int L){
    return i*M*L + j*L + k;
}

void Graph::addEdge(int src, int dest) {
    // add an edge to list adjacency from src to dest only if the edge does not already exist
   adjList[src].insert(dest);
    }



// from the mapping of the 3D points create the graph
void Graph::CreateGraph(int N,int M, int L){
    std::cout << "Creating the graph" << std::endl;
    for (int i=0; i<N; i++){
        for (int j=0; j<M; j++){
            for (int k=0; k<L; k++){
                int index = Graph::getIndex(i,j,k,N,M,L);
                if (i>0){
                    int index2 = Graph::getIndex(i-1,j,k,N,M,L);
                    addEdge(index,index2);
                }
                if (i<N-1){
                    int index2 =Graph::getIndex(i+1,j,k,N,M,L);
                    addEdge(index,index2);
                }
                if (j>0){
                    int index2 = Graph::getIndex(i,j-1,k,N,M,L);
                    addEdge(index,index2);
                }
                if (j<M-1){
                    int index2 = Graph::getIndex(i,j+1,k,N,M,L);
                    addEdge(index,index2);
                }
                if (k>0){
                    int index2 = Graph::getIndex(i,j,k-1,N,M,L);
                    addEdge(index,index2);
                }
                if (k<L-1){
                    int index2 = Graph::getIndex(i,j,k+1,N,M,L);
                    addEdge(index,index2);
                }
            }
        }
    }
}



// Performs BFS on the graph starting from vertex 'startVertex'
void Graph::BFS(int startVertex) {
    std::vector<bool> visited(adjList.size(), false);
    std::queue<int> queue;

    visited[startVertex] = true;
    queue.push(startVertex);

    while (!queue.empty()) {
        int currVertex = queue.front();
        std::cout << "Visited " << currVertex << std::endl;
        queue.pop();

        // Traverse all adjacent vertices of the current vertex
        for (int adjVertex : adjList[currVertex]) {
            if (!visited[adjVertex]) {
                visited[adjVertex] = true;
                queue.push(adjVertex);
            }
        }
    }
}


// void Graph::printGraph() {
//     std::cout << " " << std::endl;
//     int index = 0;
//     for (int i=0; i<adjList.size(); i++){
//         std::cout << "Vertex " << i << " is connected to: ";
//         for (int j=0; j<adjList[i].size(); j++){
//             std::cout << adjList[i][j] << " ";
//         }
//         std::cout << std::endl;
//     }
// }

// void Graph::printMapping() {
//     for (auto row : mapping) {
//         std::cout << "Vertex " << row.first << " is mapped to: ";
//         std::cout << row.second << " ";
//         std::cout << std::endl;
//     }
// }

int main() {
    // Create a graph given in the above diagram
    auto start = std::chrono::high_resolution_clock::now();
    Graph g(100,100,100);
    g.CreateGraph(100,100,100);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    std::cout << "Time to create the graph: " << elapsed.count() << " s\n";
    // int cpt_3 = 0;
    // int cpt_4 = 0;
    // int cpt_5 = 0;
    // int cpt_6 = 0;
    // int cpt_total = 0;
    // for (int i=0; i<g.adjList.size(); i++){
    //     if (g.adjList[i].size()==3){
    //         cpt_3++;
    //     }
    //     if (g.adjList[i].size()==4){
    //         cpt_4++;
    //     }
    //     if (g.adjList[i].size()==5){
    //         cpt_5++;
    //     }
    //     if (g.adjList[i].size()==6){
    //         cpt_6++;
    //     }
    //     cpt_total += 1;
    // }
    // std::cout << "Number of vertices with 3 neighbors: " << cpt_3 << std::endl;
    // std::cout << "Number of vertices with 4 neighbors: " << cpt_4 << std::endl;
    // std::cout << "Number of vertices with 5 neighbors: " << cpt_5 << std::endl;
    // std::cout << "Number of vertices with 6 neighbors: " << cpt_6 << std::endl;
    // std::cout << "Total number of vertices: " << cpt_total << std::endl;

    

    
}
