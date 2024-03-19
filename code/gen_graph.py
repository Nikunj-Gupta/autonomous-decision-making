import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
import room_extractor


def graph_node(substate={}): 
    nodes = list(substate.keys()) 
    print(nodes)
    nodes_dict = {} 
    for i,n in enumerate(nodes): 
        nodes_dict[i] = n 
    adjacency_matrix = np.zeros((len(nodes), len(nodes))) 
    for ni in range(len(nodes)): 
        for nj in range(len(nodes)): 
            if ni == nj: 
                continue 
            if len(substate[nodes[ni]]["rooms"]) == 1: 
                if substate[nodes[ni]]["rooms"][0] in substate[nodes[nj]]["rooms"]: 
                    adjacency_matrix[ni, nj] = 1 
            if len(substate[nodes[nj]]["rooms"]) == 1: 
                if substate[nodes[nj]]["rooms"][0] in substate[nodes[ni]]["rooms"]: 
                    adjacency_matrix[ni, nj] = 1 
            if (len(substate[nodes[ni]]["rooms"]) == 2) and (len(substate[nodes[nj]]["rooms"]) == 2): 
                if not (sorted(substate[nodes[ni]]["rooms"]) == sorted(substate[nodes[nj]]["rooms"])): 
                    if set(substate[nodes[ni]]["rooms"]) & set(substate[nodes[nj]]["rooms"]): 
                        adjacency_matrix[ni, nj] = 1 
    print(adjacency_matrix) 
    G=nx.from_numpy_array(adjacency_matrix) 
    nx.draw(G, labels=nodes_dict, with_labels=True) 
    plt.show() 

    return adjacency_matrix ,nodes_dict

def graph_adjancy(env_state):
    room_info_dict = room_extractor.room_layout(env_state)
    adj_matrix, labels = graph_node(room_info_dict["node_info"])

    return adj_matrix, labels




if __name__ == "__main__": 
    # Room 1: [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (2, 3), (3, 1), (3, 2), (3, 3), (1, 4), (2, 4), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4)]
    # Room 2: [(1, 6), (1, 7), (2, 6), (2, 7), (1, 8), (2, 8), (3, 6), (3, 7), (3, 8), (1, 9), (2, 9), (3, 9), (4, 6), (4, 7), (4, 8), (4, 9)]
    # Room 3: [(6, 1), (6, 2), (7, 1), (7, 2), (6, 3), (7, 3), (8, 1), (8, 2), (8, 3), (6, 4), (7, 4), (8, 4), (9, 1), (9, 2), (9, 3), (9, 4), (6, 5), (7, 5), (8, 5), (9, 5)]
    # Room 4: [(6, 7), (6, 8), (7, 7), (7, 8), (6, 9), (7, 9), (8, 7), (8, 8), (8, 9), (9, 7), (9, 8), (9, 9)]
    # Exit_gate (5, 3): ['Room - 1', 'Room - 3']
    # Exit_gate (2, 5): ['Room - 1', 'Room - 2']
    # Exit_gate (9, 6): ['Room - 3', 'Room - 4']
    # Agent_room (4, 3): ['Room - 1']
    # Goal_room (9, 9): ['Room - 4']
    medium_0 = {
        (5, 3): {
            "type": "Agent_room", 
            "rooms": ['Room - 1', 'Room - 3'] 
        },
        (2, 5): {
            "type": "Agent_room", 
            "rooms": ['Room - 1', 'Room - 2']
        },
        (9, 6): {
            "type": "Agent_room", 
            "rooms": ['Room - 3', 'Room - 4']
        },
        (4, 3): {
            "type": "Agent_room", 
            "rooms": ['Room - 1']
        },
        (9, 9): {
            "type": "Goal_room", 
            "rooms": ['Room - 4']
        } 
    }

    # Exit_gate (5, 3): ['Room - 1', 'Room - 2']
    # Exit_gate (5, 6): ['Room - 1', 'Room - 2']
    # Agent_room (9, 3): ['Room - 2']
    # Goal_room (9, 8): ['Room - 2'] 
    medium_1 = {
        (5, 3): {
            "type": "Exit_gate", 
            "rooms": ['Room - 1', 'Room - 2']
        }, 
        (5, 6): {
            "type": "Exit_gate", 
            "rooms": ['Room - 1', 'Room - 2']
        },
        (9, 3): {
            "type": "Agent_room", 
            "rooms": ['Room - 2'] 
        },
        (9, 8): {
            "type": "Goal_room", 
            "rooms": ['Room - 2'] 
        } 
    }
    
    # Exit_gate (8, 3): ['Room - 1', 'Room - 5']
    # Exit_gate (1, 4): ['Room - 1', 'Room - 2']
    # Exit_gate (15, 6): ['Room - 5', 'Room - 6']
    # Exit_gate (2, 8): ['Room - 2', 'Room - 3']
    # Exit_gate (12, 10): ['Room - 6', 'Room - 7']
    # Exit_gate (4, 12): ['Room - 3', 'Room - 4']
    # Exit_gate (8, 12): ['Room - 4', 'Room - 7']
    # Agent_room (14, 8): ['Room - 6']
    # Goal_room (15, 15): ['Room - 7']
    hard_0 = {
        (8, 3): {
            "type": "Exit_gate", 
            "rooms": ['Room - 1', 'Room - 5']
        }, 
        (1, 4): {
            "type": "Exit_gate", 
            "rooms": ['Room - 1', 'Room - 2']
        },
        (15, 6): {
            "type": "Exit_gate", 
            "rooms": ['Room - 5', 'Room - 6']
        },
        (2, 8): {
            "type": "Exit_gate", 
            "rooms": ['Room - 2', 'Room - 3']
        },
        (12, 10): {
            "type": "Exit_gate", 
            "rooms": ['Room - 6', 'Room - 7']
        },
        (4, 12): {
            "type": "Exit_gate", 
            "rooms": ['Room - 3', 'Room - 4']
        },
        (8, 12): {
            "type": "Exit_gate", 
            "rooms": ['Room - 4', 'Room - 7']
        },
        (14, 8): {
            "type": "Agent_room", 
            "rooms": ['Room - 6']
        },
        (15, 15): {
            "type": "Goal_room", 
            "rooms": ['Room - 7']
        } 
    }

    graph_node(medium_1) 
    # graph_node(medium_0) 
    # graph_node(hard_0) 

