from collections import deque
import gen_graph

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
  path_list = bfs_shortest_path(adj_matrix, labels)
  for i in path_list:
    path.append(labels[i])

  return path

