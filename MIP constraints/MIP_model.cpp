#include "MIP_model.hpp"
#include <iostream>
#include <tuple>


struct KeyHash {
    std::size_t operator()(const std::tuple<int, int, int>& key) const {
        const auto& [x, y, z] = key;
        return std::hash<int>()(x) ^ std::hash<int>()(y) ^ std::hash<int>()(z);
    }
};

// Constructor: initializes the number of vertices and resizes the adjacency list
Graph::Graph(int vertices) : adjList(vertices) {}


void Graph::addEdge(int src, int dest) {
    adjList[src].push_back(dest);
    adjList[dest].push_back(src); // Comment this line for a directed graph
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
