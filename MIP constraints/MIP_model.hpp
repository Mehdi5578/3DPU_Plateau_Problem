#ifndef MIP_model
#define MIP_model

#include <vector>
#include <queue>
#include <unordered_map>
#include <tuple>


    struct KeyHash {
    std::size_t operator()(const std::tuple<int, int, int>& key) const {
        return std::hash<int>()(std::get<0>(key)) 
                ^ std::hash<int>()(std::get<1>(key)) 
                ^ std::hash<int>()(std::get<2>(key));
        }
    };



class Graph {
    
public:
    Graph(int N, int M, int L);// Constructor of the size of the Graph N*M*L
    std::unordered_map<int, int> mapping; // hashing using the mapping of 3D points
    std::unordered_map<int, std::vector<int>> adjList; //adjacency list of the graph
    void addEdge(int src, int dest); // Function to add an edge to the graph
    void BFS(int startVertex); // BFS function
    void CreateMapping(int N, int M, int L); // Create the mapping of the 3D points
    void CreateGraph(int N, int M, int L); // Create the graph from the mapping of the 3D points
    void printGraph(); // Print the graph
    void printMapping(); // Print the mapping of the 3D points
    int getIndex(int i, int j, int k, int N, int M, int L); // Get the index of the 3D point
    std::tuple<int,int,int> getCoordinates(int index, int N, int M, int L); // Get the coordinates of the 3D point


};

#endif // BFS_HPP
