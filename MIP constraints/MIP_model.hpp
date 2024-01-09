#ifndef MIP_model
#define MIP_model

#include <vector>
#include <queue>

class Graph {
private:
    int numVertices; // Number of vertices in the graph
    std::vector<std::vector<int>> adjList; // Adjacency list representation

public:
    Graph(int vertices); // Constructor
    void addEdge(int src, int dest); // Function to add an edge to the graph
    void BFS(int startVertex); // BFS function
};

#endif // BFS_HPP
