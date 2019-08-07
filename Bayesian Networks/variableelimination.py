import numpy as np
import copy

def variableElimination(index, observations, model):
    """
    infering with variable elimination
    """
    
    if len(model[1][index]) <= 1:
        return [1]
    factors = []
    variables = order(model[0])
    factorvars = []
    graph = model[0]
    observedindices = [x[0] for x in observations]
    for var in variables:
        cptList = copy.deepcopy(model[1][var])
        parents = []
        for i in range(len(graph)):
            if graph[i][var] == 1:
                parents.append(i)
        args = []
        for v in [var] + parents:
            if v not in observedindices:
                args.append(v)
        pr = []
        getpr([var] + parents, observations, observedindices, cptList, pr) #get the probability as a factor of [var] + parents - observed variables
        update(pr) #remove the additional "[" and "]"
        
        pr = factorreshape(pr, args)
        factors.append(pr) #obtain the original factors, which are conditional probabilities
        factorvars.append(args) #the factors' arguments

    for var in variables:
        
        factorsupdate(var, factors, factorvars) #combine the factors with the common argument var with pointwise product
        pos = -1
        for i in range(len(factors)):
            if var in factorvars[i]:
                pos = i 

        if var != index:
            sumout(var, factors, factorvars, pos) #sum out to eliminate the argument var
    #finally only one factor is left 
    a = factors[0]
    for x in factors:
        if len(x) > len(a):
            a = x
    
    res = list(a)
    for i in range(len(res)):
        if sum(a) != 0:
            res[i] = a[i] / sum(a) #compute the conditional probabilities as result
        else:
            res[i] = 1.0 / len(a)
    return res

def factorreshape(factor, args):
    if not isinstance(factor[0],list):
        return factor
    res = np.array(factor)
    if len(args) <= 1:
        return
    for i in range(len(args)-1):
        if args[i]>args[i+1]:
            res = res.swapaxes(i, i+1)
            temp = args[i+1]
            args[i+1] = args[i]
            args[i] = temp
    
    return res.tolist()
                    
def sumout(var, factors, factorvars, index):
    """sum out to eliminate the argument var"""
    if index < 0:
        return
    factor = [[]]
    choicenum = []
    getsize([], choicenum, factors[index], factorvars[index])
    for i in range(len(choicenum)):
        if var == factorvars[index][i]:
            del choicenum[i]
            
    produceCptList(factor, choicenum)
    update(factor)
    sumoutupdate(var, factor, factors[index], factorvars[index])
    factors[index] = factor
    newvars = factorvars[index]
    while var in newvars:
        newvars.remove(var)


def sumoutupdate(var, new, old, oldvars):
    """
    recursively come to the values to be added up and add them together
    the result is added up and stored in the corresponding place of new
    """

    if not isinstance(new[0], list) and len(new) == 1:
        new[0] = sum(old)    
    elif var == oldvars[0]:
        for i in range(len(old)):
            addup(new, old[i])

    elif not isinstance(new[0], list):
        for i in range(len(new)):
            new[i] = sum(old[i])
    
    else:
        for i in range(len(new)):
            sumoutupdate(var, new[i], old[i], oldvars[1:])

def addup(new, old):
    """recursively add up and update new"""
    if not isinstance(new[0], list):
        for i in range(len(old)):
            new[i] += old[i]
    else:
        for i in range(len(old)):
            addup(new[i], old[i])   
          
def factorsupdate(var, factors, factorvars):
    """
    combine the factors with the common argument var
    """
    a = []
    for i in range(len(factors)):
        if var in factorvars[i]:
            a.append(i)
          
    if len(a) <= 1:
        return
         
    pointwise(a, factors, factorvars)
    factorsupdate(var, factors, factorvars)  
            
def pointwise(a, factors, factorvars):
    """
    pointwisely multiply factors indicated by the first 2 elems in a and update factors and factorvars"""
    newvars =[]
    choicenum = []
    getsize(newvars, choicenum, factors[a[0]], factorvars[a[0]])
    getsize(newvars, choicenum, factors[a[1]], factorvars[a[1]])
    (newvars, choicenum) = rearrange(newvars, choicenum)
    factor = [[]]   
    produceCptList(factor, choicenum)
    update(factor)
    pointwiseupdate(factor, newvars, factors[a[0]],factors[a[1]], factorvars[a[0]],factorvars[a[1]])
    factors[a[0]] = factor
    del factors[a[1]]
    factorvars[a[0]] = newvars
    del factorvars[a[1]]


def rearrange(variables, choicenum):
    newvars = list(variables)
    newvars.sort()
    newchoicenum = list(choicenum)
    for i in range(len(variables)):
        newchoicenum[newvars.index(variables[i])] = choicenum[i]
    return(newvars, newchoicenum)
   
def pointwiseupdate(factor, newvars, factors1, factors2, factorvars1, factorvars2):
    """recursively come to the values in the corresponding places and make products"""

    if newvars == []:
        return
    if (not isinstance(factors1, list)) and (not isinstance(factors2[0], list)):
        for i in range(len(factor)):
            factor[i] = factors1 * factors2[i]
        return
    elif (not isinstance(factors2, list)) and (not isinstance(factors1[0], list)):
        for i in range(len(factor)):
            factor[i] = factors1[i] * factors2
        return
    
    if factorvars1 == factorvars2 and len(factorvars1) == 1:
        for i in range(len(factor)):
            factor[i] = factors1[i] * factors2[i]
        return
                
    if newvars[0] in factorvars1 and newvars[0] not in factorvars2:
        for i in range(len(factor)):
            pointwiseupdate(factor[i], newvars[1:], factors1[i], factors2, factorvars1[1:], factorvars2)
    elif newvars[0] in factorvars1 and newvars[0] in factorvars2:
        for i in range(len(factor)):
            pointwiseupdate(factor[i], newvars[1:], factors1[i], factors2[i], factorvars1[1:], factorvars2[1:])
    else:
        for i in range(len(factor)):
            pointwiseupdate(factor[i], newvars[1:], factors1, factors2[i], factorvars1, factorvars2[1:])
        
        
        
    

def getsize(newvars, choicenum, factors, factorvars):
    """obtain the sizes of choices of areguments in choicenum and the new arguments in newvars"""

    if factorvars==[]:
        return
    if factorvars[0] not in newvars:
        newvars.append(factorvars[0])
        choicenum.append(len(factors))
    getsize(newvars, choicenum, factors[0], factorvars[1:])


def produceCptList(cpt, choicenum):
    """produce a matrix based on sizes of choices"""
    if choicenum == []:
        for i in range(len(cpt)):
            cpt[i] = 0
            
        return
    else:
        for i in range(len(cpt)):
            for j in range(choicenum[0]):
                cpt[i].append([])
            
            produceCptList(cpt[i], choicenum[1:])   
    
def getpr(variables, observations, variablesobserved, cptList, factor):
    """obtain the probabilities based on observations"""
    if variables[0] in variablesobserved:
        for x in observations:
            if x[0] == variables[0]:
                index = x[1]
        if not isinstance(cptList[0], list):
            factor.append(cptList[index])
            
        else:
            observed = list(variablesobserved)
            observed.remove(variables[0])
            getpr(variables[1:], observations, observed, cptList[index], factor)
    else:
        for i in range(len(cptList)):
            if not isinstance(cptList[0], list):
                factor.append(cptList[i])
     
            else:
                factor.append([])
                getpr(variables[1:], observations, variablesobserved, cptList[i], factor[i])
            
def update(factor):
    """
    remove the additional "[" "]"
    """
    if factor == []:
        return
    if not isinstance(factor[0], list):
        return
    if len(factor) == 1:
        for i in factor[0]:
            factor.append(i)
        del factor[0]
        update(factor)
        return            
    for i in range(len(factor)):
        if len(factor[i]) == 1:
            factor[i] = factor[i][0]
            update(factor)
        else:
            update(factor[i])
        
def order(graph):
    """
    obtain a list of vars in the oder that to be eliminated
    """
    parents = []
    n = range(len(graph))
    for i in n:
        parents.append([])
        
    for i in n:
        for j in n:
            if graph[i][j] == 1:
                parents[j].append(i)
    result = []
    while len(result) != len(graph):
        for i in n:
            if i not in result:
                if belong(parents[i], result):
                    result.append(i)
    result.reverse()
    return result

def belong(A, B):
    """
    return True if all elems in A are in B
    otherwise False
    """
    if A == []:
        return True
    
    for i in A:
        if i not in B:
            return False
    
    return True
