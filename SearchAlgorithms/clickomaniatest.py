import clickomaniaplayer as ckp
import clickomania as ck
import random


def generateState(N,M,K, initialState = None):
    """ an initial state is generated, used for testing
    initialState is a list representing the initial state
    if None, then a random initial state is generated"""
    
    if not initialState:
        initialState = list(range(N * M))
        #generate tiles row by row 
        for i in range(N*M):
            initialState[i] = random.randint(0, K)
    
    initialState = ck.State(initialState)
    initialState.fix(N,M)                     
    return initialState
        


N = 17
M = 6
K = 4
initialState = generateState(N,M,K)
print(ckp.clickomaniaPlayer(ck.Clickomania(N,M,K), initialState))
