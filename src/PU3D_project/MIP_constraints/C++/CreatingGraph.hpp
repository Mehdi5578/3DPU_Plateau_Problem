#ifndef CreatingGraph_hpp
#define CreatingGraph_hpp

#include "MIP_model.hpp"
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <tuple>


class MIP_model::Graph {
    
public:
    Graph(int N, int M, int L);// Constructor of the size of the Graph N*M*L
    std::unordered_map<int, int> mapping; // hashing using the mapping of 3D points
    std::vector<std::unordered_set<int>> adjList; //adjacency list of the graph
    void addEdge(int src, int dest); // Function to add an edge to the graph
    void BFS(int startVertex); // BFS function
    void CreateMapping(int N, int M, int L); // Create the mapping of the 3D points
    void CreateGraph(int N, int M, int L); // Create the graph from the mapping of the 3D points
    void printGraph(); // Print the graph
    void printMapping(); // Print the mapping of the 3D points
    int getIndex(int i, int j, int k, int N, int M, int L); // Get the index of the 3D point
    int countEdges(); // Count the number of edges in the graph
    std::tuple<int,int,int> getCoordinates(int index, int N, int M, int L); // Get the coordinates of the 3D point


};


#endif // BFS_HPP
