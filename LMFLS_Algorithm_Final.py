import time
from collections import defaultdict,OrderedDict
import numpy as np
from numpy import loadtxt
from sklearn.metrics.cluster import normalized_mutual_info_score
#------------------------------------------------------------------------------------------------------------
def k_shell_score(neighbors_dict):
    k_shell_scores = {node: 0 for node in neighbors_dict}
    nodes_by_degree = sorted(neighbors_dict,key=lambda node: neighbors_dict[node][1])
    for node in nodes_by_degree:
        max_neighbor_k_shell_score = max(k_shell_scores[neighbor] for neighbor in list(neighbors_dict[node][0].keys()))
        if max_neighbor_k_shell_score == 0:
            max_neighbor_k_shell_score = nodes_neighbors[node][1]
        k_shell_scores[node] = max_neighbor_k_shell_score
        nodes_neighbors.setdefault(node).append(nodes_neighbors[node][1] / (N - 1))  # degree centrality of node
        nodes_neighbors.setdefault(node).append((max_neighbor_k_shell_score))
        temp = []
        for j in list(neighbors_dict[node][0].keys()):
            temp.append(nodes_neighbors[j][1])
        nodes_neighbors.setdefault(node).append(hindex(temp, node))

def hindex(n, node):
    n = sorted(n, reverse=True)
    h = 0
    L = nodes_neighbors[node][1] #degree of node
    degCentrality = nodes_neighbors[node][2]
    for ii in range(1, len(n) + 1):
        if (n[ii - 1] + (((L / 100) + degCentrality + (ii / L)) * n[ii - 1])) < ii:
            break
        h = ii
    return h

def nodeDominance(node):
    dominanceSum = 0
    neighborss = list(nodes_neighbors[node][0].keys())
    K_Shell_improved = nodes_neighbors[node][3]+(nodes_neighbors[node][3]*nodes_neighbors[node][2])

    for ii in neighborss:
        CommonNeighbors = len(list(set(neighborss) & set(nodes_neighbors[ii][0].keys())))
        nodes_neighbors[node][0].setdefault(ii).append(CommonNeighbors)
        dominanceSum += ((nodes_neighbors[node][0][ii][0]+CommonNeighbors + 2) + (K_Shell_improved * nodes_neighbors[node][1])) / (nodes_neighbors[ii][1] + 1)
    nodes_neighbors.setdefault(i).append((float(dominanceSum) + nodes_neighbors[node][4]+K_Shell_improved))

def surrounding_communities(neighborss):
    communities = {}
    for ii in neighborss:
        if nodes_neighbors[ii][1] > 1:
            communities.setdefault(nodes_neighbors[ii][10], []).append(ii)
    return communities

def neighborCommunityScore(node,neighborss):
    sum_scores = defaultdict(int)
    Weight_Similarity = defaultdict(float)
    for neighbor in neighborss:
        if nodes_neighbors[neighbor][1] > 1:
            neighbor_group = nodes_neighbors[neighbor][10]
            sum_scores[neighbor_group] += nodes_neighbors[neighbor][6] # sum up total importance of nodes in same communities
            CommonNeighbors = nodes_neighbors[node][0][neighbor][1]*nodes_neighbors[node][0][neighbor][0]
            if nodes_neighbors[node][1] > nodes_neighbors[neighbor][1]:
                difResult = 1 + len(set(nodes_neighbors[neighbor][0].keys()).difference(set(neighborss)))
            else:
                difResult = 1 + len(set(neighborss).difference(set(nodes_neighbors[neighbor][0].keys())))
            WeightedSimilarity = CommonNeighbors / difResult
            Weight_Similarity[neighbor_group] += WeightedSimilarity

    return dict(sum_scores),dict(Weight_Similarity)
# ---------------------------------- Load Dataset -------------------------------------------
dataset_name = "karate"# name of dataset
path = "datasets/" + dataset_name + ".txt" # path to dataset

iteration1 = 2        # number of iterations for label selection step
iteration2 = 2        # number of iterations for final label selection step
threshold = 0.9
merge_flag = 1        # merge_flag=0 -> do not merge ////  merge_flag=1 -> do merge
modularity_flag = 1   # 1 means calculate modularity. 0 means do not calculate modularity
NMI_flag = 1          # 1 means calculate NMI. 0 means do not calculate NMI

# ------------------------- compute nodes neighbors and nodes degree --------------------------
nodes_neighbors = {}
i = 0
with open(path) as f:
    for line in f:
        if line.strip():
            row = str(line.strip()).split('\n')[0].split('\t')
            temp_arrey = {}
            for j in row:
                if j != '':
                    temp_arrey.setdefault(int(j),[]).append(1)
            nodes_neighbors.setdefault(i, []).append(temp_arrey)
            nodes_neighbors.setdefault(i, []).append(len(nodes_neighbors[i][0]))
        i = i+1
    N = i # number of nodes
#----------------------------------Compute node Importance -----------------
start_time = time.time()
k_shell_score(nodes_neighbors)  # k-shell score of node and H-index
for i in nodes_neighbors.keys():
    if nodes_neighbors[i][1] > 1:
        nodeDominance(i)
for i in nodes_neighbors.keys():
    if nodes_neighbors[i][1] > 1:
        counter = 0
        importanceSum = 0
        neighbors = list(nodes_neighbors[i][0].keys()) #neighbors of node i
        for j in neighbors:
            if nodes_neighbors[j][1] > 1:
                importanceSum += nodes_neighbors[j][5] #importance of neighbor of node i
                counter += 1
        if counter == 0:
            importanceAverage = 0
        else:
            importanceAverage = importanceSum / counter
        s = 0
        for j in neighbors:
            if nodes_neighbors[j][1] > 1:
                s += (nodes_neighbors[j][5] / importanceAverage) * nodes_neighbors[i][0][j][1]
        nodes_neighbors.setdefault(i).append(s+nodes_neighbors[i][5])

for i in nodes_neighbors.keys():
    if nodes_neighbors[i][1] > 1:
        temp = ((nodes_neighbors[i][0][neighbor][1], neighbor) for neighbor in list(nodes_neighbors[i][0].keys()) if (nodes_neighbors[neighbor][1] > 1))
        max_result = max(temp, default=None)

        if max_result is not None:
            if max_result[0] == 0:
                temp = ((nodes_neighbors[neighbor][6], neighbor) for neighbor in list(nodes_neighbors[i][0].keys()) if(nodes_neighbors[neighbor][1] > 1))
                max_result = max(temp, default=None)
            importantNeighbor = max_result[1]
            nodes_neighbors.setdefault(i).append(importantNeighbor) #most similar neighbor
            nodes_neighbors.setdefault(i).append(0)
            nodes_neighbors.setdefault(i).append(1)
            nodes_neighbors.setdefault(i).append(i) #initial label of node
        else:
            nodes_neighbors.setdefault(i).append(-1)
            nodes_neighbors.setdefault(i).append(1)
            nodes_neighbors.setdefault(i).append(0)
            nodes_neighbors.setdefault(i).append(i)

nodesOrder = {}
for i in nodes_neighbors.keys():
    if nodes_neighbors[i][1] >1: # degree more than 1
        if nodes_neighbors[i][9] == 1:
            nodesOrder[i] = nodes_neighbors[i][6] # total importance

nodesOrder = OrderedDict(sorted(nodesOrder.items(), key=lambda item: item[1] , reverse=True)) #sort nodes based on total importance

for i in nodesOrder.keys():
    if nodes_neighbors[i][8] == 0:
        importantNeighbor = nodes_neighbors[i][7]
        importantNeighbor_Neighbor = nodes_neighbors[importantNeighbor][7]
        if nodes_neighbors[i][6] <= nodes_neighbors[importantNeighbor][6]:
            nodes_neighbors[i][10] = importantNeighbor
        else:
            if importantNeighbor_Neighbor == i:
                nodes_neighbors[i][10] = i
                nodes_neighbors[importantNeighbor][10] = i
            elif importantNeighbor_Neighbor != i:

                if nodes_neighbors[i][6] >= nodes_neighbors[importantNeighbor_Neighbor][6]:
                    nodes_neighbors[i][10] = i
                    nodes_neighbors[importantNeighbor][10] = i
                    nodes_neighbors[importantNeighbor_Neighbor][10] = i
                else:
                    nodes_neighbors[i][10] = importantNeighbor_Neighbor
                    nodes_neighbors[importantNeighbor_Neighbor][10] = importantNeighbor_Neighbor
                    nodes_neighbors[importantNeighbor][10] = importantNeighbor_Neighbor

for i in nodesOrder.keys():
    nodes_neighbors[i][10] = nodes_neighbors[nodes_neighbors[i][10]][10]

#-------------------------Label selection step -------------------------------------
for iterr in range(iteration1):
    for i in nodesOrder.keys():
        if nodes_neighbors[i][8] == 0:
            neighborsList = list(nodes_neighbors[i][0].keys())
            groupedCommunities = surrounding_communities(neighborsList)

            if len(groupedCommunities.keys()) == 1:
                nodes_neighbors[i][10] = list(groupedCommunities.keys())[0]
                nodes_neighbors[i][8] = 1
            else:
                counts = {}
                for key in groupedCommunities:
                    counts[key] = len(groupedCommunities[key])
                labelFrequency = list(sorted(counts.items(), key=lambda item: item[1], reverse=True)) # frequency of each label
                if (labelFrequency[0][1]*threshold) >= labelFrequency[1][1]:
                    nodes_neighbors[i][10] = labelFrequency[0][0]
                else: # label influence should be computed
                    GroupScore = {}
                    neighborScore,weightedSimilarity = neighborCommunityScore(i,neighborsList) #sum of importance of labels
                    LabelFrequency = labelFrequency
                    labelInfluence = {}
                    for label in LabelFrequency:
                        labelScore = (label[1]*nodes_neighbors[label[0]][6]*neighborScore[label[0]])+weightedSimilarity[label[0]]
                        labelInfluence[label[0]] = labelScore
                    nodes_neighbors[i][10] = max(labelInfluence, key=labelInfluence.get)

#-----------------------------------final label selection----------------------------
for iterr in range(iteration2):
    for i in nodesOrder.keys():
        if nodes_neighbors[i][9] == 1:
            if nodes_neighbors[i][1] == 2:
                maxDegree = max((nodes_neighbors[n][6], n) for n in list(nodes_neighbors[i][0].keys()) if nodes_neighbors[n][1] > 1)[1]
                maxLabel = nodes_neighbors[maxDegree][10]
            else:
                key_with_max_value = max(nodes_neighbors[i][0], key=lambda x: nodes_neighbors[i][0][x][1])
                max_value = nodes_neighbors[i][0][key_with_max_value][1]
                if max_value != 0:
                    maxLabel = nodes_neighbors[key_with_max_value][10]
                else:
                    maxLabel = nodes_neighbors[i][10]
            nodes_neighbors[i][10] = maxLabel
#--------------------------------------- Merge step -------------------------
if merge_flag == 1:
    c = 0
    communityMembers = {}
    for i in nodes_neighbors.keys():
        if nodes_neighbors[i][1] > 1:
            communityMembers.setdefault(nodes_neighbors[i][10],[]).append(i)
            c += 1
    max_key = max(communityMembers, key=lambda k: len(communityMembers[k]))
    k=len(communityMembers[max_key])
    average_size = c / (len(communityMembers.keys())-1)  # average size of communities
    selectedCommunities = list({k for k, v in communityMembers.items() if len(v) <= average_size})
    selectedCommunitiesNew=[]
    idMap={}
    for r in selectedCommunities:
        temp = ((nodes_neighbors[m][6], m) for m in communityMembers[r])
        max_result = max(temp, default=None)
        selectedCommunitiesNew.append(max_result[1])
        idMap[max_result[1]]=r

    if selectedCommunitiesNew:
        for i in selectedCommunitiesNew:
            temp = []
            mergeStatus = 0
            for j in list(nodes_neighbors[i][0].keys()):
                if nodes_neighbors[j][1] > 1:

                    weightsum = sum([v[0] for v in nodes_neighbors[j][0].values()])
                    similaritySum = sum([v[1] for v in nodes_neighbors[j][0].values()])
                    temp.append((j,weightsum+similaritySum+nodes_neighbors[j][1]))
            candidate = max(temp,key=lambda x: x[1],default=(-1, -1))[0]

            if candidate != -1:
                temp = ((nodes_neighbors[neighbor][6], neighbor) for neighbor in list(nodes_neighbors[candidate][0].keys()) if (nodes_neighbors[neighbor][1] > 1) and neighbor!=i)
                max_result = max(temp, default=None)
                if max_result is not None:
                    importantNeighbor = max_result[1]

                else:
                    continue
                if nodes_neighbors[candidate][10] != nodes_neighbors[importantNeighbor][10]:
                    if nodes_neighbors[i][6] <= nodes_neighbors[nodes_neighbors[importantNeighbor][10]][6]:
                        if list(set(nodes_neighbors[i][0].keys()) & set(nodes_neighbors[nodes_neighbors[importantNeighbor][10]][0].keys())):
                            newLabel = nodes_neighbors[importantNeighbor][10]
                            mergeStatus = 1
                if mergeStatus ==1:
                    for node in communityMembers[idMap[i]]:
                        if nodes_neighbors[node][1] > 1:
                            nodes_neighbors[node][10] =newLabel
                        else:
                            nodes_neighbors[node][5] = newLabel

#------------- Assign label to nodes with degree = 1 ----------------
for i in nodes_neighbors.keys():
    if nodes_neighbors[i][1] == 1:
        if nodes_neighbors[list(nodes_neighbors[i][0].keys())[0]][1]==1:
            nodes_neighbors.setdefault(list(nodes_neighbors[i][0].keys())[0]).append(list(nodes_neighbors[i][0].keys())[0])
            nodes_neighbors.setdefault(i).append(list(nodes_neighbors[i][0].keys())[0])
        else:
            nodes_neighbors.setdefault(i).append(nodes_neighbors[list(nodes_neighbors[i][0].keys())[0]][10])
print("--- Total Execution time %s seconds ---" % (time.time() - start_time))
# ---------------------------------- Number of communities --------------------------------
nodesLabels = {}
for i in nodes_neighbors.keys():
    if nodes_neighbors[i][1] > 1:
        nodesLabels[i] = nodes_neighbors[i][10]
    else:
        nodesLabels[i] = nodes_neighbors[i][5]
number_of_communities = list(set(nodesLabels.values()))
print("Number of Communities: ", len(set(nodesLabels.values())))
# ----------------------------------- Modularity -----------------------------------------
if modularity_flag ==1:
    t = 0
    for i in nodes_neighbors:
        t = t + nodes_neighbors[i][1]
    edges = t / 2
    modu = 0
    are_neighbor = []
    for i in nodesLabels.keys():
        neighborsList = list(nodes_neighbors[i][0].keys())
        for j in nodesLabels.keys():
            if nodesLabels[i] == nodesLabels[j]:
                if nodes_neighbors[i][1] >= 1:
                    if j in neighborsList:
                        are_neighbor = 1
                    else:
                        are_neighbor = 0
                    modu = modu + (are_neighbor - ((nodes_neighbors[i][1] * nodes_neighbors[j][1]) / (2 * edges)))
    modularity_final = modu / (2 * edges)
    print('Modularity:  {}'.format(modularity_final))
# ------------------------------- NMI ----------------------------------------------
if NMI_flag == 1:
    ordered_nodes_neighbors = OrderedDict(sorted(nodesLabels.items()))
    real_labels = loadtxt("groundtruth/"+dataset_name+"_real_labels.txt", comments="#", delimiter="\t",unpack=False)
    detected_labels = []
    if dataset_name in ('karate', 'dolphins', 'polbooks', 'football'):
        for i in ordered_nodes_neighbors:
            detected_labels.append(ordered_nodes_neighbors[i])
        detected_labels = np.array(detected_labels)
        print('NMI:  {}'.format(normalized_mutual_info_score(real_labels, detected_labels)))
    else:
        nodes_map = loadtxt("datasets/nodes_map/"+dataset_name+"_nodes_map.txt", comments="#", delimiter="\t",unpack=False)
        for i in nodes_map:
            detected_labels.append(ordered_nodes_neighbors[i])
        print('NMI:  {}'.format(normalized_mutual_info_score(real_labels, detected_labels)))
#-------------------------------------------------------------------------------------

