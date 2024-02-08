# TrojanMap
Google maps replica using C++

## Class Data Structure
<p align="center"><img src="images/Class Data Structure.png" alt="Nearby" width="500"/></p>

## Item 1: Autocomplete The Location Name (Phase 1):
The purpose of this function is that whenever a user enters an incomplete name of a location, the program should output possible location names.

Example:\
Input: "Chi"\
Output: ["Chick-fil-A", "Chipotle", "Chinese Street Food"]

Obtained Output:

<p align="center"><img src="images/Item 1 Autocomplete.png" alt="Nearby" width="500"/></p>

    ->Runtime of Algorithm: O(n)
    ->Time spent: Fairly less

## Item 2-1: Find the place's coordinates in the Map (Phase 1):
This function receives the location name as an input and returns latitude & longitude coordinates corresponding to the location.

Input: "Target"\
Output: (34.0257016, -118.2843512)

Obtained output:

<p align="center"><img src="images/Item 2 Find Location.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 2 Map.png" alt="Nearby" width="500"/></p>

    ->Runtime of Algorithm: O(n)
    ->Time spent: Fairly less

## Item 2-2: Check Edit Distance Between Two Location Names (Phase 2):
This function takes two location names as input and finds the number of insert, delete or replace operations required for two strings to be similar. We have achieved this using the tabulation approach of dynamic programming.

Obtained output:

<p align="center"><img src="images/Item 2-1 Find Closest Name.png" alt="Nearby" width="500"/></p>

    ->Runtime of Algorithm: 
    ->CalculateEditDistance: O(m*n), where m = length of string 1 & n = length of string 2
    ->FindClosestName: O(k*m*n), where k = number of locations

## Item 3: Get All Categories (Phase 2):
This function returns all attributes listed in data (no duplicates).

Obtained output:

<p align="center"><img src="images/Item 3 Get all categories.png" alt="Nearby" width="500"/></p>

    ->Runtime of Algorithm: O(n*a), Where n = number of nodes (locations) & a = number of attributes.
    ->Time spent: Fairly less

## Item 4: Get All Locations In A Category (Phase 2):

In this function, the name of the attribute is passed as an input argument & all locations belonging to the input attribute are returned.

Obtained Output:

<p align="center"><img src="images/Item 4 Get All Locations of a category.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 4 Map.png" alt="Nearby" width="500"/></p>

     ->Runtime of algorithm: O(n)
     ->Time spent: Fairly less
 

## Get Locations Using A Regular Expression (Phase 2):

This function returns all location ids matching to the input [Regular Expression](https://en.wikipedia.org/wiki/Regular_expression).

    ->Runtime of Algorithm: O(n)
    ->Time spent: Average time spent on learning how regular expressions are handled in C++

## Item 6: CalculateShortestPath between two places (Phase 2):

Given two input locations, this function finds the best (shortest) route from one location to another using Dijkstra & Bellman Ford algorithm.

Obtained Output:

<p align="center"><img src="images/Item 6 Dijsktra and Bellman Ford.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 6 Map.png" alt="Nearby" width="500"/></p>

From obtained output, it can be observed that Dijkstra takes lesser time compared to Bellman Ford. This is because Bellman Ford handles a wide range of scenarios.


Comparison Table:

| Point A to Point B      | Dijkstra | Bellman Ford|
| -------------------- | ----------- |-------|
| Tommy Trojan -> Leavey |  35  | 35600 |
| Lyons Center -> PED    |  21  | 32127|
| USC Village Gym -> USC Parking |  26  | 27572|
| Subway -> kfc|  144  | 30028 |
| Chase -> City Tacos  |  45  |33604 |
| Lululemon -> Chinese Street Food  |  1  | 31252 |
| Insomnia Cookies -> Village Cobbler   |  1  | 33708 |
|USC Credit Union -> Tommy Trojan|  12  | 33466 |
| Hoover & 30th -> Hoover & 28th|  7   | 36529 |
| Food 4 Less -> Ross  |  209  | 46134 |


    ->Runtime of Dijkstra: O(V^2), Where V = Total nodes (locations) visited
    ->Runtime of Bellman Ford: O(V*E), Where V = Total nodes (locations) visited, E = Total edges between path
    ->Time spent: Good amount of time was spent, especially more on Bellman Ford

## Item 7: Cycle Detection (Phase 2):
For this function, we consider a square section of the graph & try to find if there exists a cycle path between provided subgraph coordinates.

Obtained output:

<p align="center"><img src="images/Item 7 Cycle Detection 1.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 7 map 1.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 7 Cycle Detection 2.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="img/Item 7 map 2.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 7 Cycle Detection 3.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 7 map 3.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 7 Cycle Detection 4.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 7 map 4.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 7 Cycle Detection 5.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 7 map 5.png" alt="Nearby" width="500"/></p>

 ->Runtime of Algorithm: O(n)
 ->Time spent: Average time spent on this function

## Item 8: Topological Sort (Phase 2):

Obtained output:

<p align="center"><img src="images/Item 8 Topological sort 1.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 8 Map 1.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 8 Toplogical Sort 2.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 8 Map 2.png" alt="Nearby" width="500"/></p>

    ->Runtime for algorithm: O(n)
    ->Time spent: Average amount of time was spent. The main difficulty was encountered in getting the indegree vector before performing bfs on it.

## Item 9: The Traveling Trojan Problem (AKA Traveling Salesman!) (Phase 3):

This function finds the shortest route that covers all the locations & returns back to the starting point. We’ve used Brute-Force, Brute-force enhanced with early backtracking  and 2-opt algorithm.

Obtained Output:

<p align="center"><img src="images/Item 9 Six Points.png" alt="Nearby" width="500"/></p>

Time comparison:

| Number of Nodes     | Brute Force | Backtracking | 2-opt |
| -------------------- | ----------- | ------------ | ----- |
|  6  |  0  | 0 | 0 |
|  7  |  16  | 6 | 0 |
|  12  |  time limit exceed  | ~ | ~ |


Runtime for algorithm:

    ->Brute Force: O(n!)
    ->Backtracking: O((n-1)!) 
    ->2-opt: O((n^2)*i), where n = locations to be visited, i = iterations
    ->Time spent: In 2-opt & backtracking, we encountered a bug that we were not counting distance by including first node as last one. Majority of the time from entire project was spent on these three functions.

## Item 10: Find Nearby

Given an attribute name `C`, a location name `L` and a number `r` and `k`, the app finds at most `k` locations in attribute `C` on the map near `L` with the range of `r` and return a vector of string ids. 

The order of locations are from nearest to farthest.

Obtained output:

<p align="center"><img src="images/Item 10 Find Nearby.png" alt="Nearby" width="500"/></p>

 ->Runtime of Algorithm: O(n)
 ->Time spent: Fairly less

## Item 11: Find the Shortest Path to Visit All locations (Phase 3):

This function takes a vector of location as an input argument & returns the shortest path to visit all locations.

Obtained output:

<p align="center"><img src="images/Item 11 TrojanPath.png" alt="Nearby" width="500"/></p>

<p align="center"><img src="images/Item 11 map.png" alt="Nearby" width="500"/></p>

 ->Runtime of algorithm: O((n^2)*n!*(V+E)*log(V)) where, n = Total number of nodes,  V = Number of nodes visited, E = edges between nodes
 ->Time spent: Average time was spent

## Item 12: Check the existence of the path with a constrained gas tank (Phase 3):

This function receives an input vector which has a pair of source & destination locations & gas tank capacity. Given this information, the function finds if there is a way to travel from source to destination with a given tank size, considering that we can refuel at any node.

Obtained output:

<p align="center"><img src="images/Item 12 Gas Tank.png" alt="Nearby" width="500"/></p>

     ->Runtime of algorithm: O(n*(V+E)), n = number of queries
     ->Time spent: Fairly less

## Conclusion:

By using data structures & algorithms learned through EE538, We have implemented an algorithm that can be used in any geo-spatial application for basic queries like search a location, find shortest from one location to another location & many other queries. 

## Lessons Learned:
    -> Data Structures: Graph, Tree, Priority Queue, Map, Pair
    -> Algorithms: Brute-force, 2-opt, Backtracking, Dijkstra, Bellman Ford
    -> Techniques: Dynamic programming, Backtracking, DFS, BFS
