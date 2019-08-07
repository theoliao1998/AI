import random
import copy

class State:
    """State for Clickomania graph."""
   
    def __init__(self, v, score = 0):
        self.value = list(v)
        self.score = score

    def clone(self):
        return State(self.value, self.score)

    def __repr__(self):
        """Converts an instance in string.
      
         Useful for debug or with print(state)."""
        return str(self.value)

    def __eq__(self, other):
        """Overrides the default implementation of the equality operator."""
        if isinstance(other, State):
            return self.value == other.value
        elif other==None:
            return False
        return NotImplemented

    def __hash__(self):
        """Returns a hash of an instance.
      
         Useful when storing in some data structures."""
        return hash(str(self.value))
        
    def fix(self, N, M):
        """fix the state based on rules"""
        
        for i in range(N):
            for j in range(1,M):
                if self.value[N * j + i] == 0:
                    for k in range(j):
                        self.value[N * (j-k) + i] = self.value[N * (j-k-1) + i]

                    self.value[i] = 0
                    
        for i in range(N-1):
            if self.value[N * (M-1)+i] == 0:
                for j in range(M):
                    self.value[N * j + i] = self.value[N * j + i + 1]
                    self.value[N * j + i + 1] = 0
                    
    def printState(self,N,M):        
        """visualize the state"""
        
        for i in range(N * M):
            print(self.value[i], end=' ')
            if (i % N == N - 1):
                print("")


class Clickomania:
    """Clickomania graph based on State.
    with N as length and M as height and K types of tiles"""
       
    def __init__(self, n, m, k): 
        self.N = n
        self.M = m
        self.K = k 

    def expand(self, i, succs, state, Explored):
        """recursivily find all adjoining same-type tiles and place their index in succs
        i is the index of the original tile, state is the present state, 
        explored is the list containing the index of all explored tiles
        initially, succs and Explored should be empty lists"""
        
        test = []
        if (i % self.N != 0):
            test.append(i-1) #Left
        if (i >= self.N):
            test.append(i-self.N) #Up
        if (i % self.N != self.N-1):
            test.append(i+1) #Right
        if (i < self.N*(self.M-1)):
            test.append(i+self.N) #Down
        for index in test:
            if index not in Explored:
                if state.value[index]==state.value[i]:
                    succs.add(index)
                    Explored.add(index)
                    self.expand(index, succs, state, Explored)

    def successors(self, state):
        """with a given state, return all possible states after one move
        what to be returned is a tuple, with the first element to be 1,
        and the second element to be the state"""
        
        Empty=[]  
        Explored=set()
        Succs = []
        result = []
        for i in range(self.N * self.M):
            if state.value[i] == 0:
                if state.value[i] not in Empty: 
                    Empty.append(i)
            else: 
                if i not in Explored:
                    succs=set()
                    succs.add(i)
                    Explored.add(i)
                    self.expand(i, succs, state, Explored)
                    if len(succs)>1:  
                        Succs.append(succs)               
        new_state = []
        for i in range(len(Succs)):
            new_state.append(state.clone())
        cnt = 0
        for choice in list(Succs):
            length = len(choice)
            new_state[cnt].score += (length-1)**2
            for index in choice:
                new_state[cnt].value[index]=0
            
            new_state[cnt].fix(self.N, self.M) #fix the state based on the rules  

            result.append((1,new_state[cnt])) #1 represents the path cost
            cnt += 1
            
        return result

    def isGoal(self, state):
        """if no choice of successors, the searching reaches the end"""        
        return self.successors(state) == []

