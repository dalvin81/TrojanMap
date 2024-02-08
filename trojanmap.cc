#include "trojanmap.h"


/**
 * GetLat: Get the latitude of a Node given its id. If id does not exist, return
 * -1.
 *
 * @param  {std::string} id : location id
 * @return {double}         : latitude
 */
double TrojanMap::GetLat(const std::string &id) { 
  return data[id].lat;
}

/**
 * GetLon: Get the longitude of a Node given its id. If id does not exist,
 * return -1.
 *
 * @param  {std::string} id : location id
 * @return {double}         : longitude
 */
double TrojanMap::GetLon(const std::string &id) {
  return data[id].lon;
}

/**
 * GetName: Get the name of a Node given its id. If id does not exist, return
 * "NULL".
 *
 * @param  {std::string} id : location id
 * @return {std::string}    : name
 */
std::string TrojanMap::GetName(const std::string &id) {
  return data[id].name;
}

/**
 * GetNeighborIDs: Get the neighbor ids of a Node. If id does not exist, return
 * an empty vector.
 *
 * @param  {std::string} id            : location id
 * @return {std::vector<std::string>}  : neighbor ids
 */
std::vector<std::string> TrojanMap::GetNeighborIDs(const std::string &id) {
  return data[id].neighbors;
}

/**
 * GetID: Given a location name, return the id.
 * If the node does not exist, return an empty string.
 * The location name must be unique, which means there is only one node with the name.
 *
 * @param  {std::string} name          : location name
 * @return {std::string}               : id
 */
std::string TrojanMap::GetID(const std::string &name) {
  std::string res = "";
  for(auto iter: data) {
    if(iter.second.name == name) {
      res = iter.first;
      break;
    }
  }
  return res;
}

/**
 * GetPosition: Given a location name, return the position. If id does not
 * exist, return (-1, -1).
 *
 * @param  {std::string} name          : location name
 * @return {std::pair<double,double>}  : (lat, lon)
 */
std::pair<double, double> TrojanMap::GetPosition(std::string name) {
  std::pair<double, double> results(-1, -1);
  std::map<std::string, Node>::iterator it;
  for(auto it:data) {
    if(it.second.name==name) {
      results.first = it.second.lat;
      results.second = it.second.lon;
    }
  }
  return results;
}

/**
 * CalculateEditDistance: Calculate edit distance between two location names
 * @param  {std::string} a          : first string
 * @param  {std::string} b          : second string
 * @return {int}                    : edit distance between two strings
 */
int TrojanMap::CalculateEditDistance(std::string a, std::string b) {     
  std::transform(a.begin(), a.end(), a.begin(), ::tolower);
  std::transform(b.begin(), b.end(), b.begin(), ::tolower);
  if (a==b){
    return 0;
  }
  int m = a.size(); //rows
  int n = b.size(); //cols
  std::vector<std::vector<int>>dp_mat(m+1,std::vector<int>(n+1,0));
  //first row
  for (int i=0;i<=n;i++){
    dp_mat[0][i]=i;
  }
  //first col
  for (int i=1;i<=m;i++){
    dp_mat[i][0]=i;
  }
  for (int i=1;i<=m;i++){
    for (int j=1;j<=n;j++){
      if (a[i-1]==b[j-1]){
        dp_mat[i][j]=dp_mat[i-1][j-1];
      }
      else{
        dp_mat[i][j]=1+std::min(dp_mat[i-1][j],std::min(dp_mat[i-1][j-1],dp_mat[i][j-1]));
      }
    }
  }

  return dp_mat[m][n];
}


/**
 * FindClosestName: Given a location name, return the name with the smallest edit
 * distance.
 *
 * @param  {std::string} name          : location name
 * @return {std::string} tmp           : the closest name
 */
std::string TrojanMap::FindClosestName(std::string name) {
  int minDistance = INT_MAX;
  std::string closestName;

  // Iterate through all location names in the map
  // Use a set to avoid duplicates
  std::unordered_set<std::string> locationNames;
  for (auto it:data) {
    locationNames.insert(it.second.name);
  }

  for (const std::string& locationName : locationNames) {
    int distance = CalculateEditDistance(name, locationName);

    if (distance < minDistance) {
      minDistance = distance;
      closestName = locationName;
    }
  }

  return closestName;
}

/**
 * Autocomplete: Given a parital name return all the possible locations with
 * partial name as the prefix. The function should be case-insensitive.
 *
 * @param  {std::string} name          : partial name
 * @return {std::vector<std::string>}  : a vector of full names
 */
std::vector<std::string> TrojanMap::Autocomplete(std::string name) {
  std::vector<std::string> results;
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  for(auto x:data) {
    std::string lowercaseLocation = x.second.name;
    std::transform(lowercaseLocation.begin(), lowercaseLocation.end(), lowercaseLocation.begin(), ::tolower);
    if (lowercaseLocation.find(name) == 0) {
            results.push_back(x.second.name);
    }
  }
  return results;
}

/**
 * GetAllCategories: Return all the possible unique location categories, i.e.
 * there should be no duplicates in the output.
 *
 * @return {std::vector<std::string>}  : all unique location categories
 */
std::vector<std::string> TrojanMap::GetAllCategories() {
  std::set<std::string> uniqueCategories;
  std::vector<std::string> categories;

  for(auto it:data) {
    for(auto category:it.second.attributes) {
      uniqueCategories.insert(category);
    }
  }
  for(auto x:uniqueCategories) {
    categories.push_back(x);
  }
  return categories;
}

/**
 * GetAllLocationsFromCategory: Return all the locations of the input category (i.e.
 * 'attributes' in data.csv). If there is no location of that category, return
 * (-1, -1). The function should be case-insensitive.
 *
 * @param  {std::string} category         : category name (attribute)
 * @return {std::vector<std::string>}     : ids
 */
std::vector<std::string> TrojanMap::GetAllLocationsFromCategory(std::string category) {
  std::vector<std::string> res;
  std::transform(category.begin(), category.end(), category.begin(), ::tolower);
  for(auto it:data) {
    if(it.second.attributes.find(category)!=it.second.attributes.end()) {
      res.push_back(it.first);
    }
  }
  return res;
}

/**
 * GetLocationRegex: Given the regular expression of a location's name, your
 * program should first check whether the regular expression is valid, and if so
 * it returns all locations that match that regular expression.
 *
 * @param  {std::regex} location name      : the regular expression of location
 * names
 * @return {std::vector<std::string>}     : ids
 */
std::vector<std::string> TrojanMap::GetLocationRegex(std::regex location) {
  std::vector<std::string> res;
  for(auto it:data) {
    if(std::regex_match(it.second.name, location)) {
      res.push_back(it.first);
    }
  }
  return res;
}

/**
 * CalculateDistance: Get the distance between 2 nodes.
 * We have provided the code for you. Please do not need to change this function.
 * You can use this function to calculate the distance between 2 nodes.
 * The distance is in mile.
 * The distance is calculated using the Haversine formula.
 * https://en.wikipedia.org/wiki/Haversine_formula
 * 
 * @param  {std::string} a  : a_id
 * @param  {std::string} b  : b_id
 * @return {double}  : distance in mile
 */
double TrojanMap::CalculateDistance(const std::string &a_id,
                                    const std::string &b_id) {
  // Do not change this function
  Node a = data[a_id];
  Node b = data[b_id];
  double dlon = (b.lon - a.lon) * M_PI / 180.0;
  double dlat = (b.lat - a.lat) * M_PI / 180.0;
  double p = pow(sin(dlat / 2), 2.0) + cos(a.lat * M_PI / 180.0) *
                                           cos(b.lat * M_PI / 180.0) *
                                           pow(sin(dlon / 2), 2.0);
  double c = 2 * asin(std::min(1.0, sqrt(p)));
  return c * 3961;
}

/**
 * CalculatePathLength: Calculates the total path length for the locations
 * inside the vector.
 * We have provided the code for you. Please do not need to change this function.
 * 
 * @param  {std::vector<std::string>} path : path
 * @return {double}                        : path length
 */
double TrojanMap::CalculatePathLength(const std::vector<std::string> &path) {
  // Do not change this function
  double sum = 0;
  for (int i = 0; i < int(path.size()) - 1; i++) {
    sum += CalculateDistance(path[i], path[i + 1]);
  }
  return sum;
}

/**
 * CalculateShortestPath_Dijkstra: Given 2 locations, return the shortest path
 * which is a list of id. Hint: Use priority queue.
 *
 * @param  {std::string} location1_name     : start
 * @param  {std::string} location2_name     : goal
 * @return {std::vector<std::string>}       : path
 */

struct pathData {
  double distance;
  std::vector<std::string> path;
};


std::vector<std::string> TrojanMap::CalculateShortestPath_Dijkstra(
    std::string location1_name, std::string location2_name) {
  std::vector<std::string> res;

  //retrieve source & destination id
  std::string src_id = GetID(location1_name);
  std::string dest_id = GetID(location2_name);

  //priority queue that will store updated distance of neighboring vertices
  std::priority_queue<std::pair<double, std::string>, std::vector<std::pair<double, std::string>>, std::greater<std::pair<double, std::string>>> distances;
  
  //storing visited node id & path information
  std::unordered_map<std::string, pathData> pathInfo;

  distances.push(std::make_pair(0, src_id));      //store source location id as visited
  pathInfo[src_id] = {0, {src_id}};               //initialize path with the source location

  //run loop till there are no unvisited nodes
  while (!distances.empty()) {
    auto [curr_distance, curr_node] = distances.top();  //choosing node with minimum distance
    distances.pop();

    if (curr_node == dest_id) {
      break;
    }
    //update neighboring vertices difference
    for (const auto &neighbor : data[curr_node].neighbors) {
      double dist = pathInfo[curr_node].distance + TrojanMap::CalculateDistance(curr_node, neighbor);

      if (pathInfo.find(neighbor) == pathInfo.end() || pathInfo[neighbor].distance > dist) {
        pathInfo[neighbor] = {dist, pathInfo[curr_node].path};
        pathInfo[neighbor].path.push_back(neighbor);
        distances.push(std::make_pair(dist, neighbor));
      }
    }
  }
  //check if path to destination exists
  if (pathInfo.find(dest_id) != pathInfo.end()) {
    res = pathInfo[dest_id].path;
  }
  return res;
}

/**
 * CalculateShortestPath_Bellman_Ford: Given 2 locations, return the shortest
 * path which is a list of id. Hint: Do the early termination when there is no
 * change on distance.
 *
 * @param  {std::string} location1_name     : start
 * @param  {std::string} location2_name     : goal
 * @return {std::vector<std::string>}       : path
 */
std::vector<std::string> TrojanMap::CalculateShortestPath_Bellman_Ford( std::string location1_name, std::string location2_name) {

  std::vector<std::string> path;

  //retrieve start and end location IDs
  std::string start_id=TrojanMap::GetID(location1_name);
  std::string end_id=TrojanMap::GetID(location2_name);

  //stores shortest path start node to each node
  std::unordered_map<std::string,std::vector<std::string>> pathInfo;

  //stores updated distance during each iteartion from source to each node
  std::unordered_map<std::string,double>currDist_map;

  //stores previous iteration's distance in order to compare with current iteration
  std::unordered_map<std::string,double>prevDistances;

  pathInfo.insert(std::make_pair(start_id, std::vector<std::string>()));    //insert starting location with distance = 0
  prevDistances.insert(std::make_pair(start_id, 0));

  bool flag = true; //a flag to monitor if there are changes in distances as we go to next relaxation
  double tempDist;
  while(flag) {
    for(auto &node:prevDistances) {
      for(auto &neighbor:data[node.first].neighbors){
        //check if distance of neighboring node is already present in distance map
        if(currDist_map.find(neighbor) == currDist_map.end()) {
          currDist_map[neighbor] = INT_MAX;
        }
        //also check if present in previous iteration's distance map
        if(prevDistances.find(neighbor) == prevDistances.end()) {
          tempDist = INT_MAX;
        } else {
          tempDist = prevDistances[neighbor];
        }
        //calculating new neighbor distance & if it is smaller then update
        double newDist = prevDistances[node.first] + TrojanMap::CalculateDistance(node.first, neighbor);
        currDist_map[neighbor] = std::min(currDist_map[neighbor], std::min(tempDist, newDist));
        if(currDist_map[neighbor] == newDist) {
          std::vector<std::string>updatedPath = pathInfo[node.first];
          updatedPath.push_back(node.first);
          if(pathInfo.find(neighbor) == pathInfo.end()) {
            pathInfo.insert(std::make_pair(neighbor, updatedPath));
          } else {
            pathInfo[neighbor] = updatedPath;
          }
        }
      }
    }
    //check in any change in distances from previous iteration
    if(currDist_map == prevDistances) {
      flag = false;
    } else {
      prevDistances = currDist_map;
    }
  }
  //check if path to destination exists
  if(pathInfo.find(end_id) != pathInfo.end()) {
    path = pathInfo[end_id];
    path.push_back(end_id);
  }
  return path;
}

/**
 * Traveling salesman problem: Given a list of locations, return the shortest
 * path which visit all the places and back to the start point.
 *
 * @param  {std::vector<std::string>} input : a list of locations needs to visit
 * @return {std::pair<double, std::vector<std::vector<std::string>>} : a pair of total distance and the all the progress to get final path, 
 *                                                                      for example: {10.3, {{0, 1, 2, 3, 4, 0}, {0, 1, 2, 3, 4, 0}, {0, 4, 3, 2, 1, 0}}},
 *                                                                      where 10.3 is the total distance, 
 *                                                                      and the first vector is the path from 0 and travse all the nodes and back to 0,
 *                                                                      and the second vector is the path shorter than the first one,
 *                                                                      and the last vector is the shortest path.
 */
// Please use brute force to implement this function, ie. find all the permutations.
std::pair<double, std::vector<std::vector<std::string>>> TrojanMap::TravelingTrojan_Brute_force(
                                    std::vector<std::string> location_ids) {
  std::vector<std::vector<std::string>> result;
        double minDistance = std::numeric_limits<double>::infinity();

        // Generate all permutations of location_ids.
        do {
            // Ensure the route starts and ends at the same location.
            std::vector<std::string> currentRoute = location_ids;
            currentRoute.push_back(location_ids.front());

            // Calculate the distance of the current route.
            double currentDistance = CalculatePathLength(currentRoute);

            // Update the minimum distance and the corresponding route.
            if (currentDistance < minDistance) {
                minDistance = currentDistance;
                result.push_back(currentRoute);
            }

        } while (std::next_permutation(location_ids.begin() + 1, location_ids.end()));

        // Return the result.
        return std::make_pair(minDistance, result);
}

//Helper function for backtracking
void TrojanMap::Backtrack(std::vector<std::string> &location_ids, std::pair<double, std::vector<std::vector<std::string>>> &results, 
std::vector<std::string> currRoute, double &min_dist, double dist){
  //if currRoute has all the locations, calculate the total distance
  if(currRoute.size() == location_ids.size()){
    //Latest distance is distance from last point of currRoute and the new node added
    double last_dist = CalculateDistance(currRoute[currRoute.size() - 1], location_ids[0]);
    dist += last_dist;
    currRoute.push_back(location_ids[0]);

    //If the new calculated distance is less than the minimum distance, add the new node to the currRoute and update the minimum distance
    if(dist < min_dist){
      results.first = dist;
      results.second.push_back(currRoute);
      min_dist = dist;
    }

    //Else remove the new node as it does not lead to minimum distance
    currRoute.erase(currRoute.end() - 1);
    dist -= last_dist;
    return;
  }
  else{ //if currRoute does not have all the locations, add the next location_id and continue the Backtracking
    for(size_t i = 0; i < location_ids.size(); i++){
      if(find(currRoute.begin(), currRoute.end(), location_ids[i]) == currRoute.end()){
        double next_dist = CalculateDistance(currRoute[currRoute.size() - 1], location_ids[i]);
        dist += next_dist;
        currRoute.push_back(location_ids[i]);
        Backtrack(location_ids, results, currRoute, min_dist, dist);
        currRoute.erase(currRoute.end() - 1);
        dist -= next_dist;
      }
    }
    return;
  }
}

// Please use backtracking to implement this function
std::pair<double, std::vector<std::vector<std::string>>> TrojanMap::TravelingTrojan_Backtracking(
                                    std::vector<std::string> location_ids) {
    std::vector<std::string> currRoute;
  std::pair<double, std::vector<std::vector<std::string>>> results;
  currRoute.push_back(location_ids[0]);
  double INF = DBL_MAX;
  Backtrack(location_ids, results, currRoute, INF, 0);
  
  return results;
}

void TrojanMap::reverse_2opt(std::vector<std::string> &location_ids,int i,int j){
  std::reverse(location_ids.begin()+i+1,location_ids.begin()+j+1);
}

double TrojanMap::calculateCost(std::vector<std::string> &location_ids, int i, int j, int n) {
  return -TrojanMap::CalculateDistance(location_ids[i], location_ids[(i + 1) % n])
       - TrojanMap::CalculateDistance(location_ids[j], location_ids[(j + 1) % n]) + TrojanMap::CalculateDistance(location_ids[i], location_ids[j])
       + TrojanMap::CalculateDistance(location_ids[(i + 1) % n], location_ids[(j + 1) % n]);
}

// Hint: https://en.wikipedia.org/wiki/2-opt
std::pair<double, std::vector<std::vector<std::string>>> TrojanMap::TravelingTrojan_2opt(
      std::vector<std::string> location_ids) {
    std::pair<double, std::vector<std::vector<std::string>>> records;
        int n = location_ids.size();
        double currDistance = TrojanMap::CalculatePathLength(location_ids)+TrojanMap::CalculateDistance(location_ids[0], location_ids[n-1]);
        records.first = currDistance;
        location_ids.push_back(location_ids[0]);
        records.second.push_back(location_ids);
        location_ids.pop_back();
        double newDistance;
        bool flag = true;

        
        while(flag) {
          flag = false;
          for(int i = 0; i <= n-2; i++) {
            for(int j = i+1; j <= n-1; j++) {
              if ((i==0 && j!=n-1 && j!=1) || (i!=0 && j!=i-1 && j!=i+1)) {
                newDistance = TrojanMap::calculateCost(location_ids, i, j, n);

                if(newDistance < 0) {
                  currDistance = newDistance;
                  TrojanMap::reverse_2opt(location_ids, i, j);
                  records.first += newDistance;
                  location_ids.push_back(location_ids[0]);
                  records.second.push_back(location_ids);
                  location_ids.pop_back();
                  flag = true;

              }
              
            }
          }
        }
      }
    return records;
}

// This is optional
std::pair<double, std::vector<std::vector<std::string>>> TrojanMap::TravelingTrojan_3opt(
      std::vector<std::string> location_ids){
  std::pair<double, std::vector<std::vector<std::string>>> records;
  return records;
}

/**
 * Given CSV filename, it read and parse locations data from CSV file,
 * and return locations vector for topological sort problem.
 * We have provided the code for you. Please do not need to change this function.
 * Example: 
 *   Input: "topologicalsort_locations.csv"
 *   File content:
 *    Name
 *    Ralphs
 *    KFC
 *    Chick-fil-A
 *   Output: ['Ralphs', 'KFC', 'Chick-fil-A']
 * @param  {std::string} locations_filename     : locations_filename
 * @return {std::vector<std::string>}           : locations
 */
std::vector<std::string> TrojanMap::ReadLocationsFromCSVFile(
    std::string locations_filename) {
  std::vector<std::string> location_names_from_csv;
  std::fstream fin;
  fin.open(locations_filename, std::ios::in);
  std::string line, word;
  getline(fin, line);
  while (getline(fin, word)) {
    location_names_from_csv.push_back(word);
  }
  fin.close();
  return location_names_from_csv;
}

/**
 * Given CSV filenames, it read and parse dependencise data from CSV file,
 * and return dependencies vector for topological sort problem.
 * We have provided the code for you. Please do not need to change this function.
 * Example: 
 *   Input: "topologicalsort_dependencies.csv"
 *   File content:
 *     Source,Destination
 *     Ralphs,Chick-fil-A
 *     Ralphs,KFC
 *     Chick-fil-A,KFC
 *   Output: [['Ralphs', 'Chick-fil-A'], ['Ralphs', 'KFC'], ['Chick-fil-A', 'KFC']]
 * @param  {std::string} dependencies_filename     : dependencies_filename
 * @return {std::vector<std::vector<std::string>>} : dependencies
 */
std::vector<std::vector<std::string>> TrojanMap::ReadDependenciesFromCSVFile(
    std::string dependencies_filename) {
  std::vector<std::vector<std::string>> dependencies_from_csv;
  std::fstream fin;
  fin.open(dependencies_filename, std::ios::in);
  std::string line, word;
  getline(fin, line);
  while (getline(fin, line)) {
    std::stringstream s(line);
    std::vector<std::string> dependency;
    while (getline(s, word, ',')) {
      dependency.push_back(word);
    }
    dependencies_from_csv.push_back(dependency);
  }
  fin.close();
  return dependencies_from_csv;
}

/**
 * DeliveringTrojan: Given a vector of location names, it should return a
 * sorting of nodes that satisfies the given dependencies. If there is no way to
 * do it, return a empty vector.
 *
 * @param  {std::vector<std::string>} locations                     : locations
 * @param  {std::vector<std::vector<std::string>>} dependencies     : prerequisites
 * @return {std::vector<std::string>} results                       : results
 */
std::vector<std::string> TrojanMap::DeliveringTrojan(std::vector<std::string> &location_names,
                                          std::vector<std::vector<std::string>> &dependencies) {
    std::unordered_map<std::string, std::vector<std::string>> graph;
    std::unordered_map<std::string, int> in_degree;

    // Construct the directed graph and calculate in-degrees
    for (const auto &dependency : dependencies) {
        graph[dependency[0]].push_back(dependency[1]);
        in_degree[dependency[1]]++;
    }

    // Perform topological sort
    std::queue<std::string> q;
    for (const auto &location : location_names) {
        if (in_degree[location] == 0) {
            q.push(location);
        }
    }

    std::vector<std::string> result;
    while (!q.empty()) {
        std::string current = q.front();
        q.pop();
        result.push_back(current);

        for (const auto &neighbor : graph[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    // Check if a feasible route exists
    if (result.size() != location_names.size()) {
        std::cout << "No feasible route exists." << std::endl;
        return {};
    }

    return result;
}

/**
 * inSquare: Give a id retunr whether it is in square or not.
 *
 * @param  {std::string} id            : location id
 * @param  {std::vector<double>} square: four vertexes of the square area
 * @return {bool}                      : in square or not
 */
bool TrojanMap::inSquare(std::string id, std::vector<double> &square) {
  double latitude = TrojanMap::GetLat(id);
  double longitude = TrojanMap::GetLon(id);
  if((latitude < square[2] && latitude > square[3]) && (longitude < square[1] && longitude > square[0])) {
    return true;
  }
  return false;
}


/**
 * GetSubgraph: Give four vertexes of the square area, return a list of location
 * ids in the squares
 *
 * @param  {std::vector<double>} square         : four vertexes of the square
 * area
 * @return {std::vector<std::string>} subgraph  : list of location ids in the
 * square
 */
std::vector<std::string> TrojanMap::GetSubgraph(std::vector<double> &square) {
  // include all the nodes in subgraph
  std::vector<std::string> subgraph;
  for(auto it: data) {
    if(TrojanMap::inSquare(it.first, square)) {
      subgraph.push_back(it.first);
    }
  }
  return subgraph;
}

bool TrojanMap::detection_helper(std::string &node,std::set<std::string>& visited,std::string& parent,std::vector<double> &square){
  //mark current visited
  visited.insert(node);

  //run dfs on node's neighbor
  for (auto neighbor:data[node].neighbors){
    //check neighbor in square
    if (TrojanMap::inSquare(neighbor, square)){
      //check neighbor not visited
      if (visited.find(neighbor)==visited.end()){
        if (TrojanMap::detection_helper(neighbor,visited,node,square)){
          return true;
        }
      }
      else{
        //already neighbor was visited
        if (neighbor!=parent){
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * Cycle Detection: Given four points of the square-shape subgraph, return true
 * if there is a cycle path inside the square, false otherwise.
 *
 * @param {std::vector<std::string>} subgraph: list of location ids in the
 * square
 * @param {std::vector<double>} square: four vertexes of the square area
 * @return {bool}: whether there is a cycle or not
 */
bool TrojanMap::CycleDetection(std::vector<std::string> &subgraph, std::vector<double> &square) {
  std::set<std::string>visited;
  std::string temp_parent = "None";
  for (auto node:subgraph){
    if (visited.find(node) == visited.end()){
      //dfs for unvisited nodes
      if (TrojanMap::detection_helper(node, visited, temp_parent, square)) {
        //cycle detected
        return true;
      }
    }
  }
  return false;
}

/**
 * FindNearby: Given a class name C, a location name L and a number r,
 * find all locations in class C on the map near L with the range of r and
 * return a vector of string ids
 *
 * @param {std::string} className: the name of the class
 * @param {std::string} locationName: the name of the location
 * @param {double} r: search radius
 * @param {int} k: search numbers
 * @return {std::vector<std::string>}: location name that meets the requirements
 */
std::vector<std::string> TrojanMap::FindNearby(std::string attributesName, std::string name, double r, int k) {

  //retrieve all locations belonging to attributesName
  std::vector<std::string> possibleLocations= TrojanMap::GetAllLocationsFromCategory(attributesName);
  
  std::priority_queue<std::pair<double,std::string>> queue; 
  
  std::string main_loc=TrojanMap::GetID(name);

  double temp;

  for (auto loc:possibleLocations){
    //avoid location passed in argument in result
    if (loc!= main_loc){
      temp =TrojanMap::CalculateDistance(loc, main_loc);
      //only include locations within r miles
      if (temp < r) {
        //only include k number of locations
        if (queue.size() < k) {
          queue.push(std::make_pair(temp, loc));
        } else {
          //if distance is lesser than first element in queue then swap
          if (queue.top().first > temp){
            queue.pop();
            queue.push(std::make_pair(temp, loc));
          }
        }
      }
    }
  }
  std::vector<std::string> res(queue.size());
  for (int i = queue.size()-1; i >= 0; i--) {
    res[i] = queue.top().second;
    queue.pop();
  }
  
  return res;
}


/**
 * Shortest Path to Visit All Nodes: Given a list of locations, return the shortest
 * path which visit all the places and no need to go back to the start point.
 *
 * @param  {std::vector<std::string>} input : a list of locations needs to visit
 * @return {std::vector<std::string> }      : the shortest path
 */
double TrojanMap::CalculateAdjDistance(std::string &id1,std::string &id2,std::map<std::pair<std::string,std::string>,double> &adj_dis){
    return adj_dis[{id1,id2}];
}

void TrojanMap::TSPBacktrack(std::vector<std::string> &current_path,
std::set<std::string> &visited,std::vector<std::string> &location_ids,
std::vector<std::string> &min_path, double &min_dist,
std::map<std::pair<std::string,std::string>,double> &adj_dis){
        //check if all nodes are visited
        if (visited.size() == location_ids.size()) {
            double temp_dist = 0;
            for (size_t i = 0; i < current_path.size() - 1; ++i) {
                temp_dist += TrojanMap::CalculateAdjDistance(current_path[i], current_path[i + 1], adj_dis);
            }
            if (temp_dist < min_dist) {
                min_path = current_path;
                min_dist = temp_dist;
            }
            return;
        }

        //recursively check for minimum distance path
        for (const auto &location_id : location_ids) {
            if (visited.find(location_id) == visited.end()) {
                //action
                visited.insert(location_id);
                current_path.push_back(location_id);

                //recursion
                TrojanMap::TSPBacktrack(current_path, visited, location_ids, min_path, min_dist, adj_dis);

                //backtrack
                visited.erase(location_id);
                current_path.pop_back();
            }
        }
  }

/**
 * Shortest Path to Visit All Nodes: Given a list of locations, return the shortest
 * path which visit all the places and back to the start point.
 *
 * @param  {std::vector<std::string>} input : a list of locations needs to visit
 * @return {std::pair<double, std::vector<std::vector<std::string>>} : a pair of
 * total distance and the all the progress to get final path
 */
std::vector<std::string> TrojanMap::TrojanPath(std::vector<std::string> &location_names) {
    std::vector<std::string> res;
        std::vector<std::string> location_ids;

        // Convert location names to location ids
        for (const auto &name : location_names) {
            location_ids.push_back(TrojanMap::GetID(name));
        }

        std::map<std::pair<std::string, std::string>, double> adj_distance;
        std::map<std::pair<std::string,std::string>,std::vector<std::string>> adj_path;

        //build matrix that stores distnce & shortest route for adjacent nodes 
        for (size_t i = 0; i < location_ids.size(); ++i) {
            for (size_t j = 0; j < location_ids.size(); ++j) {
                if (i != j) {
                    std::vector<std::string> temp_path = TrojanMap::CalculateShortestPath_Dijkstra(location_names[i], location_names[j]);
                    adj_path.insert({{location_ids[i],location_ids[j]},temp_path});
                    adj_distance[{location_ids[i], location_ids[j]}] = TrojanMap::CalculatePathLength(temp_path);
                }
            }
        }

        double min_dist = std::numeric_limits<double>::max();
        std::vector<std::string> temp_path;
        std::vector<std::string> min_path;
        std::set<std::string> visited;

        TrojanMap::TSPBacktrack(temp_path, visited, location_ids, min_path, min_dist, adj_distance);

        //constructing final route
        for (size_t i = 0; i < location_ids.size() - 1; ++i) {
            res.insert(res.end(), adj_path[{min_path[i],min_path[i+1]}].begin(), adj_path[{min_path[i],min_path[i+1]}].end());
            //avoid duplicate locations in consecutive paths
            if (i != location_ids.size() - 2) {
                res.pop_back(); 
            }
        }

        return res;
    }
/**
 * Given a vector of queries, find whether there is a path between the two locations with the constraint of the gas tank.
 *
 * @param  {std::vector<std::pair<double, std::vector<std::string>>>} Q : a list of queries
 * @return {std::vector<bool> }      : existence of the path
 */
bool TrojanMap::isNodeExist(const std::string &node) {
        return data.find(node) != data.end();
    }

bool TrojanMap::canTravel(const std::string &start, const std::string &destination, double gasTankSize) {
    if (!(TrojanMap::isNodeExist(start)) || !(TrojanMap::isNodeExist(destination))) {
        return false;
    }

    std::unordered_set<std::string> visited;
    std::queue<std::pair<std::string, double>> bfsQueue;

    bfsQueue.push({start, gasTankSize});
    visited.insert(start);

    while (!bfsQueue.empty()) {
        auto current = bfsQueue.front();
        bfsQueue.pop();

        for (const auto &neighborId : data[current.first].neighbors) {
            double distance = TrojanMap::CalculateDistance(current.first, neighborId);
            if (distance <= current.second && visited.find(neighborId) == visited.end()) {
                if (neighborId == destination) {
                    return true; 
                }

                bfsQueue.push({neighborId, gasTankSize});
                visited.insert(neighborId);
            }
        }
    }

    return false;
}


std::vector<bool> TrojanMap::Queries(const std::vector<std::pair<double, std::vector<std::string>>>& q) {
    std::vector<bool> ans;
    for(const auto &query:q) {
      double gasTankSize = query.first;
      const std::vector<std::string> &path = query.second;

      std::string start_id = TrojanMap::GetID(path[0]);
      std::string dest_id = TrojanMap::GetID(path[1]);

      bool result = TrojanMap::canTravel(start_id, dest_id, gasTankSize);
      ans.push_back(result);
      
    }
    return ans;
}

/**
 * CreateGraphFromCSVFile: Read the map data from the csv file
 * We have provided the code for you. Please do not need to change this function.
 */
void TrojanMap::CreateGraphFromCSVFile() {
  // Do not change this function
  std::fstream fin;
  fin.open("src/lib/data.csv", std::ios::in);
  std::string line, word;

  getline(fin, line);
  while (getline(fin, line)) {
    std::stringstream s(line);

    Node n;
    int count = 0;
    while (getline(s, word, ',')) {
      word.erase(std::remove(word.begin(), word.end(), '\''), word.end());
      word.erase(std::remove(word.begin(), word.end(), '"'), word.end());
      word.erase(std::remove(word.begin(), word.end(), '{'), word.end());
      word.erase(std::remove(word.begin(), word.end(), '}'), word.end());
      if (count == 0)
        n.id = word;
      else if (count == 1)
        n.lat = stod(word);
      else if (count == 2)
        n.lon = stod(word);
      else if (count == 3)
        n.name = word;
      else {
        word.erase(std::remove(word.begin(), word.end(), ' '), word.end());
        if (isalpha(word[0])) n.attributes.insert(word);
        if (isdigit(word[0])) n.neighbors.push_back(word);
      }
      count++;
    }
    data[n.id] = n;
  }
  fin.close();
}