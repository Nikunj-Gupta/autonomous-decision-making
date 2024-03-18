import matplotlib.pyplot as plt
import networkx as nx
import rooms

class Node:
    def __init__(self, name):
        self.parent_room = []
        self.exit_node = name
        self.child_room = []

    def add_parrent(self, child):
        self.parent_room.append(child)
    
    def add_child(self, child):
        self.child_room.append(child)

def find_exits(obstacles):

    current_exits = []  # Track exit patterns of the current room
    for x, y in obstacles:# Check for vertical exit pattern "# . #"
        if (x, y) in obstacles and (x, y + 1) not in obstacles and (x, y + 2) in obstacles:
            current_exits.append((x, y + 1))
    # Check for horizontal exit pattern "#.#"
        if (x, y) in obstacles and (x + 1, y) not in obstacles and (x + 2, y) in obstacles:
            current_exits.append((x + 1, y))
    
    return current_exits

def cartesianNeighbor(x, y):
    return [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1,y), (x+1, y+1)]

def cartesianVH(x, y):
    return [(x-1, y), (x, y-1), (x, y+1), (x+1,y)]

def adjacent_room(data, exit):

    exit_rooms = {}

    for x,y in exit:
        exit_rooms[(x, y)] = []
        lists = cartesianVH(x,y)
        for room_number, subdata in data.items():
            for item in lists:
                if item in subdata:
                    exit_rooms[(x,y)].append("Room - " + str(room_number))
                    
    return exit_rooms

def split_into_rooms(data, room_number=1, sublists=None):

    if sublists is None:
        sublists = {}
    while data:
        current_point = data.pop(0)
        current_sublist = [current_point]
        i = 0
        while i < len(current_sublist):
            point = current_sublist[i]
            neighbors = cartesianNeighbor(*point)
            for neighbor in neighbors:
                if neighbor in data:
                    current_sublist.append(neighbor)
                    data.remove(neighbor)
            i += 1
        sublists[room_number] = current_sublist
        room_number += 1
    return sublists


# Usage:
env = rooms.load_env(f"layouts/medium_0.txt", f"medium_o.mp4")
agent_pos = env.agent_position
# print(agent_pos)
goal_pos = env.goal_position
# print(goal_pos)
empty_points = env.occupiable_positions
# print(empty_points)
empty_points.append(goal_pos)
# print(empty_points)
wall_point = env.obstacles
exits = find_exits(wall_point)
points = [item for item in empty_points if item not in exits]

# Split the data into sublists
map_rooms = split_into_rooms(points.copy())

# Print the sublists
for room_number, sublist in map_rooms.items():
    print(f"Room {room_number}: {sublist}")

room_way = adjacent_room(map_rooms, exits)

for exit_g, rooms_c in room_way.items():
    print(f"Exit_gate {exit_g}: {rooms_c}")

agent_room = {}
goal_room = {}

for room_number, (room_name, room_points) in enumerate(map_rooms.items(), start=1):
    if agent_pos in room_points:
        agent_room.setdefault(agent_pos, []).append("Room - " + str(room_number))
    if goal_pos in room_points:
        goal_room.setdefault(goal_pos, []).append("Room - " + str(room_number))

for A, B in agent_room.items():
    print(f"Agent_room {A}: {B}")

for A, B in goal_room.items():
    print(f"Goal_room {A}: {B}")