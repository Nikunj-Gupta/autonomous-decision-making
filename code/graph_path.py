from collections import deque
from math import sqrt
import gen_graph
import rooms
import heapq

def shortest_path(adj_matrix, node_list):
    # Initialize variables
    # print(node_list)
    node_to_index = {node: i for i, node in enumerate(node_list.keys())}  # Map nodes to indices

    source = node_to_index[len(node_list) - 2]
    goal = node_to_index[len(node_list) - 1]

    n = len(adj_matrix)
    distance = [float('inf')] * n
    distance[source] = 0
    visited = [False] * n
    previous = [-1] * n

    priority_queue = [(0, source)]  # Priority queue (distance, node)

    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)

        if current_node == goal:
            # Reconstruct path
            path = []
            while current_node != -1:
                path.append(node_list[current_node])
                current_node = previous[current_node]
            return path[::-1]

        if visited[current_node]:
            continue

        visited[current_node] = True

        for neighbor, weight in enumerate(adj_matrix[current_node]):
            if weight != 0 and not visited[neighbor]:
                new_distance = distance[current_node] + weight
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    return None  # No path found

def bfs_shortest_path(adj_matrix, node_list):

  # Initialize variables
  node_to_index = [i for i, node in enumerate(node_list)]  # Map nodes to indices

  source = node_to_index[-2]
  goal = node_to_index[-1]

  n = len(adj_matrix)  # Number of vertices
  visited = [False] * n  # Keeps track of visited vertices
  parent = [-1] * n  # Stores parent of each vertex for path reconstruction

  # Create a queue for BFS traversal
  queue = deque([source])
  visited[source] = True

  # BFS traversal
  while queue:
    u = queue.popleft()
    # Check if goal is reached
    if u == goal:
      # Reconstruct the path using parent pointers
      path = []
      while u != -1:
        path.append(u)
        u = parent[u]
      return path[::-1]  # Reverse to get the path from source to goal

    # Explore adjacent vertices
    for v in range(n):
      if adj_matrix[u][v] and not visited[v]:
        visited[v] = True
        parent[v] = u  # Store parent for path reconstruction
        queue.append(v)

  # No path found
  return None


def goal_path(envior_state):
  
  path = []
  adj_matrix, labels = gen_graph.graph_adjancy(envior_state)
  # path_list = bfs_shortest_path(adj_matrix, labels)
  # for i in path_list:
  #   path.append(labels[i])
  path_list = shortest_path(adj_matrix, labels)

  return path_list

if __name__ == "__main__": 
    # Print the resulting structure
    # print(combined_data)

    env = rooms.load_env(f"layouts/hard_1.txt")
    path = []
    adj_matrix, labels = gen_graph.graph_adjancy(env)
    path_list = shortest_path(adj_matrix, labels)


    path_list_1 = bfs_shortest_path(adj_matrix, labels)
    for i in path_list_1:
      path.append(labels[i])

    print(path_list)
    print(path)