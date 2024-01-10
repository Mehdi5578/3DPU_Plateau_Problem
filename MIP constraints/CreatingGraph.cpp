#include "CreatingGraph.hpp"
#include <iostream>
#include <tuple>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <unordered_set>
#include <chrono>




// Constructor: initializes the number of vertices and resizes the adjacency list
MIP_model::Graph::Graph(int N, int M, int L) : adjList(N * M * L) {

}




std::tuple<int,int,int> MIP_model::Graph::getCoordinates(int index, int N, int M, int L){
    int i = index/(M*L);
    int j = (index-i*M*L)/L;
    int k = index-i*M*L-j*L;
    return std::make_tuple(i,j,k);
}


int MIP_model::Graph::getIndex(int i, int j, int k, int N, int M, int L){
    return i*M*L + j*L + k;
}

void MIP_model::Graph::addEdge(int src, int dest) {
    // add an edge to list adjacency from src to dest only if the edge does not already exist
   adjList[src].insert(dest);
    }



// from the mapping of the 3D points create the graph
void MIP_model::Graph::CreateGraph(int N,int M, int L){
    std::cout << "Creating the graph" << std::endl;
    for (int i=0; i<N; i++){
        for (int j=0; j<M; j++){
            for (int k=0; k<L; k++){
                int index = MIP_model::Graph::getIndex(i,j,k,N,M,L);
                if (i>0){
                    int index2 = MIP_model::Graph::getIndex(i-1,j,k,N,M,L);
                    addEdge(index,index2);
                }
                if (i<N-1){
                    int index2 =MIP_model::Graph::getIndex(i+1,j,k,N,M,L);
                    addEdge(index,index2);
                }
                if (j>0){
                    int index2 = MIP_model::Graph::getIndex(i,j-1,k,N,M,L);
                    addEdge(index,index2);
                }
                if (j<M-1){
                    int index2 = MIP_model::Graph::getIndex(i,j+1,k,N,M,L);
                    addEdge(index,index2);
                }
                if (k>0){
                    int index2 = MIP_model::Graph::getIndex(i,j,k-1,N,M,L);
                    addEdge(index,index2);
                }
                if (k<L-1){
                    int index2 = MIP_model::Graph::getIndex(i,j,k+1,N,M,L);
                    addEdge(index,index2);
                }
            }
        }
    }
}



// Performs BFS on the graph starting from vertex 'startVertex'
void MIP_model::Graph::BFS(int startVertex) {
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


// void MIP_model::Graph::printGraph() {
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

// void MIP_model::Graph::printMapping() {
//     for (auto row : mapping) {
//         std::cout << "Vertex " << row.first << " is mapped to: ";
//         std::cout << row.second << " ";
//         std::cout << std::endl;
//     }
// }

