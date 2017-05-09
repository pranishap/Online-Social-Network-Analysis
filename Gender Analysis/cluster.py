"""
cluster.py
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pickle

def count_friends(users):
    
    c = Counter()
    for user in users:
        friends = user['followers']
        c.update(friends)
    return c

def create_graph(users, friend_counts,friendsList):
    graph = nx.Graph()
    list = []
    
    for user in users:
        graph.add_node(user['screen_name'])
        followers = user['followers']
        for follower in followers:
            graph.add_edge(follower,user['screen_name'])
    return graph

def can_remove(nodeA,nodeB,graphCopy):
    nodetoEdgeDict = get_dict(graphCopy)
    edges_for_nodeA = nodetoEdgeDict[nodeA]
    edges_for_nodeB = nodetoEdgeDict[nodeB]
    if(len(edges_for_nodeA) > 1 and len(edges_for_nodeB) > 1):
        return 1
    else:
        return 0

def get_dict(graphCopy):
    nodetoEdgeDict = {}
    for node in graphCopy.nodes():
        if(node not in nodetoEdgeDict.keys()):
            edgeList = []
            edgeList.extend(graphCopy.edges([node]))
            nodetoEdgeDict[node] = edgeList
        else:
            edgeList = nodetoEdgeDict[node]
            edgeList.extend(graphCopy.edges([node]))
            nodetoEdgeDict[node] = edgeList
    return nodetoEdgeDict


def partition_girvan_newman(graph):
    graphCopy = graph.copy()
    betweenness = nx.edge_betweenness_centrality(graphCopy)
    sortedBetweeness = sorted(betweenness.items(),key=lambda x: (-x[1],x[0]))
    for edge in sortedBetweeness:
        val = can_remove(edge[0][0],edge[0][1],graphCopy)
        if(val == 1) :
            graphCopy.remove_edge(edge[0][0],edge[0][1])
        if(nx.number_connected_components(graphCopy) > 5):
            break
    components = list(nx.connected_component_subgraphs(graphCopy))
    return graphCopy,components

def draw_network(graph, users, filename,friendsList):
    
    labels = {}
    for user in users:
        labels[user['screen_name']] = user['screen_name']
        followers = user['followers']
    for commonFriend in friendsList:
        labels[commonFriend] = commonFriend
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title('Graph', fontsize=12)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_size=100,node_color='#f08080',edge_color='#c5c5c5',labels = labels,with_labels = True, font_size=10)
    plt.tight_layout()
    plt.savefig(filename, format="PNG")



def main():
    users = pickle.load(open('users.pkl', 'rb'))
    friendsList = pickle.load(open('friendList.pkl', 'rb'))
    friend_counts = count_friends(users)
    graph = create_graph(users, friend_counts,friendsList)
    draw_network(graph, users, 'networkBeforePartition.png',friendsList)
    graphCopy,clusters = partition_girvan_newman(graph)
    draw_network(graphCopy, users, 'networkAfterPartition.png',friendsList)
    pickle.dump(graphCopy, open('graph.pkl', 'wb'))



if __name__ == '__main__':
    main()

