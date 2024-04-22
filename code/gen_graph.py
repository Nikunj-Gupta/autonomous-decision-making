import numpy as np 
import room_extractor
from math import sqrt


def graph_node(substate={}):
    nodes = list(substate.keys()) 
    nodes_dict = {} 
    for i,n in enumerate(nodes): 
        nodes_dict[i] = n 
    adjacency_matrix = np.zeros((len(nodes), len(nodes))) 
    for ni in range(len(nodes)): 
        for nj in range(len(nodes)): 
            if ni == nj: 
                continue 
            if len(substate[nodes[ni]]["rooms"]) == 1: 
                # print(substate[nodes[ni]]["rooms"][0])
                if substate[nodes[ni]]["rooms"][0] in substate[nodes[nj]]["rooms"]: 
                    x1, y1 = nodes[ni]
                    x2, y2 = nodes[nj]
                    # Calculate the distance using the distance formula
                    distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    adjacency_matrix[ni, nj] = distance 

            if len(substate[nodes[nj]]["rooms"]) == 1: 
                if substate[nodes[nj]]["rooms"][0] in substate[nodes[ni]]["rooms"]: 
                    x1, y1 = nodes[ni]
                    x2, y2 = nodes[nj]
                    # Calculate the distance using the distance formula
                    distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    adjacency_matrix[ni, nj] = distance 
                    
            if (len(substate[nodes[ni]]["rooms"]) == 2) and (len(substate[nodes[nj]]["rooms"]) == 2): 
                if not (sorted(substate[nodes[ni]]["rooms"]) == sorted(substate[nodes[nj]]["rooms"])): 
                    if set(substate[nodes[ni]]["rooms"]) & set(substate[nodes[nj]]["rooms"]): 
                        x1, y1 = nodes[ni]
                        x2, y2 = nodes[nj]
                        # Calculate the distance using the distance formula
                        distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        adjacency_matrix[ni, nj] = distance 

    return adjacency_matrix ,nodes_dict

def graph_adjancy(env_state):
    room_info_dict = room_extractor.room_layout(env_state)
    adj_matrix, labels = graph_node(room_info_dict["node_info"])

    return adj_matrix, labels
