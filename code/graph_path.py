from collections import deque

def bfs_shortest_path(adj_matrix, source, goal):

  # Initialize variables
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

# Example usage
adj_matrix = [
  [0, 0, 1, 1],
  [0, 0, 1, 1],
  [1, 1, 0, 1],
  [1, 1, 0, 0]
]
source = 2
goal = 3

shortest_path = bfs_shortest_path(adj_matrix, source, goal)

if shortest_path:
  print("Shortest path from", source, "to", goal, ":", shortest_path)
else:
  print("No path exists from", source, "to", goal)