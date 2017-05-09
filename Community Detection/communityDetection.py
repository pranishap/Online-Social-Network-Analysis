
from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request

def example_graph():
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):
    node2distances = defaultdict(int)
    node2num_paths = defaultdict(int)
    node2parents = defaultdict(list)
    visited = {}
    q = deque()
    q.append(root)
    node2distances[root] = 0
    distance = 0
    while (len(q) !=0 ):
            v = q.popleft()
            visited[v] = 'True'
            if(node2distances[v] <  max_depth):
                for u in graph.neighbors(v):
                    if(u not in visited):
                        distance = node2distances[v] + 1
                        if(u not in node2distances):
                            node2distances[u] = distance
                            node2parents[u].append(v)
                            q.append(u)
                        else:
                            if(distance == node2distances[u]):
                                node2parents[u].append(v)

    node2num_paths[root]= 1
    for key in node2parents.keys():
        parents = node2parents[key]
        length = len(parents)
        node2num_paths[key]= length

    return node2distances,node2num_paths,node2parents

def complexity_of_bfs(V, E, K):
    return V + E


def bottom_up(root, node2distances, node2num_paths, node2parents):
    edge2value = defaultdict(int)
    newNode2num_paths = defaultdict(int)
    visited = set()
    parentList = node2parents.values()
    parentSet = set()
    for pList in parentList:
        for p in pList:
            parentSet.add(p)
    allNodes = set(node2num_paths.keys())
    leafnodes = allNodes.difference(parentSet)
    q = deque()
    for leaf in leafnodes:
        node2num_paths[leaf] = 1
        q.append(leaf)
    while len(q) !=0:
        node = q.popleft()
        visited.add(node)
        parentsOfnode = node2parents[node]
        sumOfParents = 0
        for parent in parentsOfnode:
            sumOfParents = sumOfParents + node2num_paths[parent]
        for parent in parentsOfnode:
            if(node in newNode2num_paths.keys()):
                edgeValue = (node2num_paths[parent] * newNode2num_paths[node]) / sumOfParents
            else:
                edgeValue = (node2num_paths[parent] * node2num_paths[node]) / sumOfParents
            if(parent not in list(q)):
                q.append(parent)
            if(parent not in newNode2num_paths.keys()):
                newNode2num_paths[parent] = 1 + edgeValue
            else:
                newNode2num_paths[parent] = newNode2num_paths[parent] + edgeValue
            
            ls = [parent,node]
            ls.sort()
            key = ls[0],ls[1]
            visited.add(parent)
            edge2value[key] = edgeValue

    return edge2value


def approximate_betweenness(graph, max_depth):
    betweenness = defaultdict(int)
    for root in graph.nodes():
        node2distances,node2num_paths,node2parents = bfs(graph, root, max_depth)
        edge2value = bottom_up(root, node2distances, node2num_paths, node2parents)
        for edge in edge2value.keys():
            betweenness[edge] = betweenness[edge] + edge2value[edge]

    for edge in betweenness.keys():
        betweenness[edge] = round((betweenness[edge] / 2) , 1)
    return betweenness



def is_approximation_always_right():
    return "no"


def partition_girvan_newman(graph, max_depth):
    graphCopy = graph.copy()
    betweenness = approximate_betweenness(graphCopy, max_depth)
    sortedBetweeness = sorted(betweenness.items(),key=lambda x: (-x[1],x[0]))
    for edge in sortedBetweeness:
        graphCopy.remove_edge(edge[0][0],edge[0][1])
        if(nx.number_connected_components(graphCopy) > 1):
            break
    components = list(nx.connected_component_subgraphs(graphCopy))
    return components

def get_subgraph(graph, min_degree):
    graphCopy = graph.copy()
    degreeDict = graphCopy.degree(graphCopy.nodes())
    for node in degreeDict.keys():
        if(degreeDict[node] < min_degree):
            graphCopy.remove_node(node)
    return graphCopy

def volume(nodes, graph):
    numOfEdges = 0
    for edge in graph.edges():
        if((edge[0] in nodes) or (edge[1] in nodes) ):
            numOfEdges = numOfEdges + 1
    return numOfEdges



def cut(S, T, graph):
    numOfEdges = 0
    for edge in graph.edges():
        if (edge[0] in S and edge[1] in T):
                numOfEdges = numOfEdges + 1
        elif(edge[0] in T and edge[1] in S):
                numOfEdges = numOfEdges + 1
    return numOfEdges




def norm_cut(S, T, graph):
    ncvS = cut(S, T, graph) / volume(S, graph)
    ncvT = cut(S, T, graph) / volume(T, graph)
    return ncvS + ncvT


def score_max_depths(graph, max_depths):
    tuppleList = []
    for depth in max_depths:
        component = partition_girvan_newman(graph, depth)
        value = norm_cut(component[0], component[1], graph)
        tuple = depth,value
        tuppleList.append(tuple)
    return tuppleList


def make_training_graph(graph, test_node, n):
    graphCopy = graph.copy()
    neighbors = sorted(graphCopy.neighbors(test_node))
    for i in range(n):
        graphCopy.remove_edge(neighbors[i],test_node)

    return graphCopy




def jaccard(graph, node, k):
    neighbors = set(graph.neighbors(node))
    scores = []
    for n in graph.nodes():
        if(n not in neighbors and n != node):
            neighbors2 = set(graph.neighbors(n))
            tupple = node , n
            scores.append((tupple, 1. * len(neighbors & neighbors2) / len(neighbors | neighbors2)))

    newlist = sorted(scores, key=lambda x: (-x[1],x[0]))
    return newlist[:k]



def path_score(graph, root, k, beta):
    scores = []
    neighbors = set(graph.neighbors(root))
    paths = nx.shortest_path_length(graph, root)
    v = max(paths, key=paths.get)
    node2distances,node2num_paths,node2parents = bfs(graph, root, paths[v])
    for node in graph.nodes():
        if(node not in neighbors and node != root):
            length = node2distances[node]
            value = (beta ** length) * node2num_paths[node]
            tupple = root , node
            scores.append((tupple , value))
    newlist = sorted(scores, key=lambda x: (-x[1],x[0]))
    return newlist[:k]


def evaluate(predicted_edges, graph):
    edgeList = list(nx.generate_edgelist(graph , data=False))
    counter = 0
    for pEdge in predicted_edges:
        p1 = pEdge[0] + " " + pEdge[1]
        p2 = pEdge[1] + " " + pEdge[0]
        if(p1 in edgeList or p2 in edgeList):
            counter = counter + 1
    return counter / len(predicted_edges)




def download_data():
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')


def read_graph():
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():
    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
      (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())
          
    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
                (train_graph.order(), train_graph.number_of_edges()))
          
          
    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
                evaluate([x[0] for x in jaccard_scores], subgraph))
          
    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
                evaluate([x[0] for x in path_scores], subgraph))


if __name__ == '__main__':
    main()
