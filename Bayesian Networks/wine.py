import csv
import structurelearning as sl
import variableelimination as ve


def accScore(D, model, var, varchoices, observedindices = None):
    if observedindices == None:
        observedindices = range(len(D[0]))
    count = 0
    for line in D:
        if varchoices[var][best(line, model, var, varchoices, observedindices)] == line[var]:
            count += 1
    
    return count / len(D) 

def best(line, model, var, varchoices, observedindices):
    observations = []
    for i in observedindices:
        if i!= var:
            observations.append((i,varchoices[i].index(line[i])))       
    prob = ve.variableElimination(var, observations, model) 
    return prob.index(max(prob))
    
def mostrelatedvars(D, model, var, varchoices):
    variables = list(range(len(D[0])))
    variables.remove(var)
    accscores = [0 for v in variables]
    mostrelated = [[] for v in variables]
    CheckAllSubsets(variables, accscores, mostrelated, D, model, var, varchoices)
    return(accscores, mostrelated)
    
def CheckAllSubsets(variables, accscores, mostrelated, D, model, var, varchoices): 

    n = len(variables)
    for i in range(2**n): 
        subset = []

        for j in range(n): 
            if(i >> j ) % 2 == 1: 
                subset.append(variables[j])
        
        if subset == []:
            continue
            
        score = accScore(D, model, var, varchoices, subset)
        if score > accscores[len(subset) - 1]:
            accscores[len(subset) - 1] = score
            mostrelated[len(subset) - 1] = subset
            print("k = ",len(subset)," : ", subset," score: ", score)

    
    


with open('wine.csv', 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    reader = list(reader)
    data = reader[1:]
    for i in range(len(data)):
        data[i] = list(map(int, data[i]))
        
    D = data[4746:]
    data = data[:4746]
    graph = [0,0]
    cptList = [0,0]
    score = [0,0]

    (graph[0], score[0]) = sl.K2Algorithm(11, data, sl.K2Score) 
    #graph[0] = [[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    cptList[0] = sl.MLEstimationVariable(graph[0], data) 
    #graph[1] = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    (graph[1], score[1]) = sl.K2Algorithm(11, data, sl.BICScore) 
    cptList[1] = sl.MLEstimationVariable(graph[1], data)
    varchoices = []
    for i in range(len(data[0])):
        varchoices.append(sl.getrange(i,data)[0])
   
    
    print("Graph0: ",graph[0])
    print(score[0])
    
    print("Graph1: ",graph[1])  
    print(score[1])
    
    for i in range(len(graph)):
        print("case ", i)
        print("acc score: ",accScore(D,(graph[i], cptList[i]),11,varchoices))
        (scores, related) = mostrelatedvars(D, (graph[i], cptList[i]), 11, varchoices)
        #print("related: ", related, " score: ", scores)
