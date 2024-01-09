#ifndef MIP_model
#define MIP_model

#include <vector>
#include <queue>

class Graph {
    
public:
    Graph(int vertices); // Constructor
    std::unordered_map<std::tuple<int, int, int>, int> mapping; // hashing using the mapping of 3D points
    std::unordered_map<int, std::vector<int>> adjList; //adjacency list of the graph
    void addEdge(int src, int dest); // Function to add an edge to the graph
    void BFS(int startVertex); // BFS function
};

#endif // BFS_HPP
