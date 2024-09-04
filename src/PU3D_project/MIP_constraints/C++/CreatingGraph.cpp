#include "CreatingGraph.hpp"
#include <iostream>
#include <tuple>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
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



// count the  number of edges in the graph
int MIP_model::Graph::countEdges(){
    int cpt = 0;
    for (int i=0; i<adjList.size(); i++){
        cpt += adjList[i].size();
    }
    return cpt/2;
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


void MIP_model::Graph::printGraph() {
    std::cout << " " << std::endl;
    int index = 0;
    for (int i=0; i<adjList.size(); i++){
        std::cout << "Vertex " << i << " is connected to: ";
        for (int j : adjList[i]){
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}

void MIP_model::Graph::printMapping() {
    for (auto row : mapping) {
        std::cout << "Vertex " << row.first << " is mapped to: ";
        std::cout << row.second << " ";
        std::cout << std::endl;
    }
}


// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>

int main() {
    // Open the text file
    std::ifstream file("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/tests/list.txt");
    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return 1;
    }

    std::vector<int> my_vector;
    std::string line;

    // Read each line from the file
    while (getline(file, line)) {
        // Convert the line to an int and add to the vector
        std::cout << line << std::endl;
        std::cout << typeid(line).name() << std::endl;
        my_vector.push_back(std::stoi(line));
    }

    // Use the vector as needed
    for (int num : my_vector) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}


