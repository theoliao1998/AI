import math
import time
import random
import numpy as np

#structure learning

def K2Algorithm(K, data, scoreFunction, timelimit = 1800):
    """
    apply K2ALgorithm with a time limit
    constantly check the randomly produced orders 
    applied if the number of the possible values is big
    K is the maximum number of parents of one variable
    timelimit is by default 1800s = 30min
    """
    graph = []
    maxTotalScore = None
    start = time.time()
    order = list(range(len(data[0])))
    timecost = 0
    while (timecost < timelimit):
        (g,score) = K2Ordered(K, data, scoreFunction,order)
        random.shuffle(order)
        if maxTotalScore == None or maxTotalScore < score:
            graph, maxTotalScore = g, score
        end = time.time()
        timecost = end - start
    return [graph, maxTotalScore]    


def K2AlgorithmAll(K, data, scoreFunction):
    """check all order, applied if the number of possible orders is samll""" 
    graph = [0]
    maxTotalScore =[]
    n = list(range(len(data[0])))
    perm(n, 0, len(n), graph, maxTotalScore, K, data, scoreFunction)
    return [graph[0], maxTotalScore[0]]

    
def perm(n,begin,end, graph, maxTotalScore, K, data, scoreFunction):
    """
    recursivly produce all permutations and check that order
    """
    if begin>=end:
        (g,score) = K2Ordered(K,data, scoreFunction, list(n))
        if maxTotalScore == []:
            maxTotalScore.append(score)
            graph[0] = g
        elif score > maxTotalScore[0]:
            graph[0],maxTotalScore[0] = g,score           
    else:
        i=begin
        for j in range(begin,end):
            n[j],n[i]=n[i],n[j]
            perm(n,begin+1,end, graph, maxTotalScore, K, data, scoreFunction)
            n[j],n[i]=n[i],n[j]

def K2Ordered(K, data, scoreFunction, order = None):
    """
    K2 algorithm implementation with a given order
    K is the maximum number of parents
    the order is by default the original order
    """
    if order == None:
        order = range(len(data[0]))
    graph = []
    n = len(data[0])
    for i in range(n):
        graph.append([])
        for j in range(n):
            graph[i].append(0)
    
    totalscore = 0
    
    for i in order:
        pi = []
        score = scoreFunction(i, pi, data)
        proceed = True
        while(proceed and len(pi)<K):

            (score, proceed) = maxNewScore(i, pi, data, scoreFunction, score, order)
        
        totalscore += score
        
        for j in pi:
            graph[j][i] = 1
    
    return (graph, totalscore)

def K2Score(variable, parents, data):
    """the K2 creterion score function for structure learning, use ln value"""
    (choice,r) = getrange(variable, data)
    q = 1
    p_choice = []
    for x in parents:
        (y,z) = getrange(x,data)
        q *= z
        p_choice.append(y)
    
    N = []
    score = 0
    for i in range(q):
        N.append([])
        for j in range(r):
            N[i].append(0)

    for j in range(q):
        sumN = 0
        right = 0  
       
        for k in range(r):
            N[j][k] = count(variable, parents, data, choice[k], parent_choice(p_choice, j))
            right += math.log(math.factorial(N[j][k]))
            sumN += N[j][k]
        
        bignumber = math.factorial(sumN + r -1)
        left = math.log(math.factorial(r-1)) - math.log(bignumber)
        score += (left + right)
                
    return score
    
def BICScore(variable, parents, data):
    """Bayesian Information Criterion score function for structure learning"""
    (choice,r) = getrange(variable, data)
    q = 1
    p_choice = []
    for x in parents:
        (y,z) = getrange(x,data)
        q *= z
        p_choice.append(y)
    
    N = []
    right = - q * (r-1) * math.log(len(data))
    
    for i in range(q):
        N.append([])
        for j in range(r):
            N[i].append(0)
    
    for j in range(q):
        left = 0
        sumN = 0
        
        for k in range(r):
            N[j][k] = count(variable, parents, data, choice[k], parent_choice(p_choice, j))
            sumN += N[j][k]
        
        for k in range(r):
            if N[j][k] != 0:
                left += N[j][k] * math.log(N[j][k]/sumN)
    
    left *= 2
    score = left + right
            
    return score 
        
def maxNewScore(variable, pi, data, scoreFunction, score_old, order):
    """update the parents pi to maximize score, 
    return the maximized new score and a bool representing whether it's updated"""
    
    score_new = score_old
    index_new = -1
    new = False
    for j in order:
        if j == variable:
            break
        if j not in pi:
            score = scoreFunction(variable, pi + [j], data)
            if (score > score_new):
                score_new = score
                index_new = j
    
    if index_new >= 0:
        pi.append(index_new)
        new = True
        
    return (score_new, new)
        

def getrange(variable, data):
    """Used to return r_i and obtain the range of possible values of X_i"""
    
    Min = Max = data[0][variable]
    for x in [line[variable] for line in data]:
        if x < Min:
            Min = x
        if x > Max:
            Max = x
    choice = list(range(Min, Max+1))
    return (choice, len(choice))  
        
def parent_choice(p_choice, j):
    """given the list of lists represnting possible choices of parents, 
    return the j-th possible choice of the combinational parents' values
    """
     
    result = list(range(len(p_choice)))
    sizes = [1]
    for i in range(len(p_choice)):
        sizes.append(sizes[i] * len(p_choice[i])) #sizes store the acculated product of the first i sizes
    
    for i in reversed(range(len(p_choice))):
        index = j // sizes[i]  #the choice of the i-th variable
        result[i] = p_choice[i][index] 
        j %= sizes[i] 
    
    return result;
     
########################################################################
#Parameter learning

def MLEstimationVariable(graph, data):
    """
    maximum likehood algorithm to estimate the conditional probabilities of the whole graph
    """
    
    parents = []
    n = range(len(graph))
    for i in n:
        parents.append([])
    
    for i in n:
        for j in n:
            if graph[i][j] == 1:
                parents[j].append(i)

    cpt = []
    for i in n:
        cpt.append(MLEstimation(i, parents[i], data))
    
    return cpt
        
def MLEstimation(variable, parents, data):
    """
    maximum likehood algorithm to estimate the conditional probabilities of a variable
    """
    (choice,r) = getrange(variable, data) #obtain the possible choices of variable
    
    if parents == []: #if there's no parent, directly obtion the 1-D cpt
        cptList = []
        count = 0
        for i in range(r):
            cptList.append(0)
            
        for line in data:
            count += 1
            cptList[choice.index(line[variable])] += 1
            
        for i in range(r):
            if cptList[i] != 0:
                cptList[i] /= count
        
        return cptList

    p_choice = [] #the possible choices of parents
    for i in parents:
        p_choice.append(getrange(i,data)[0])
    
    cptList = []
    for i in range(r):
        cptList.append([])
    
    produceCptList(cptList, p_choice) #produce the empty matrix 
    countAll = [[]]
    produceCptList(countAll, p_choice)
    countAll = countAll[0]
    
    for line in data:
        values = [line[i] for i in parents]
        indices = [p_choice[i].index(values[i]) for i in range(len(values))]
        updatecount(countAll, indices) #update counting in the corresponding place
        index = choice.index(line[variable])
        updatecount(cptList, [index] + indices)
    
    for i in range(r):
        getpr(cptList[i], countAll, r) #obtain the probabilities based on the countings 
    
    return cptList
    
    
    
def produceCptList(cpt, p_choice):
    """
    produce a matrix based on the sizes given
    p_choice is a list of lists, each list element corresponds to a dimension 
    """
    if p_choice == []:
        for i in range(len(cpt)):
            cpt[i] = 0
            
        return
    else:
        for i in range(len(cpt)):
            for j in p_choice[0]:
                cpt[i].append([])
            
            produceCptList(cpt[i], p_choice[1:])
            
def updatecount(counttable, indices):
    """update the counting with the choices given in indices"""
    
    if len(indices)==1:
        counttable[indices[0]] += 1
        return
    else:
        updatecount(counttable[indices[0]], indices[1:])
        
def getpr(prtable, counttable, choicenum):
    """obtain the probabilities based on the countings"""
    if isinstance(prtable[0],int):
        for i in range(len(prtable)):
            if prtable[i] != 0:
                prtable[i] /= counttable[i]
            elif counttable[i] == 0: #if such a case never happens in training, the pr is assumed to be equal for every choice of variable
                prtable[i] = 1.0 / choicenum
    
    else:
        for i in range(len(prtable)):
            getpr(prtable[i], counttable[i], choicenum)


def count(variable, parents, data, choice, parent_choice):
    """count the number of N_{ijk}"""
    n = 0
    for line in data:
        if line[variable] == choice:            
            parent_value = [line[i] for i in parents]
            if parent_choice == parent_value:
                n += 1
    
    return n 
    
