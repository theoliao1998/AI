import clickomania as ck
import random
import sys
import math

sys.setrecursionlimit(100000)  # set the maximum recursion depth
            
def MCTS(ValuedGraph, state, budget = 100):
    """Monte Carlo Tree Search for clickomznia
    search for the best next state"""
    
    MAXROUNDNUM = 500; #The max round number for each simulation, make sure that the simulation is terminated within limited steps, otherwise it might be too slow to make a decision
    
    class MCTSNode:
        def __init__(self, State = None, Parent = None, Path_Cost = 1):
            self.parent = Parent
            self.children = [] #the scuccessors that has been expanded
            self.n = 0         #count of visit times 
            self.Q = 0.0       #the value of score of this node
            self.state = State
            if self.parent==None:
                self.path_cost = 0
            else:
                self.path_cost = Path_Cost + Parent.path_cost
        
        def addChild(self, node):
            self.children.append(node)
            
        def isAllExpanded(self):
            """check whether all the successors of this node has been expanded"""
            expanded_states = set([ele.state.value for ele in self.children])
            for choice in [choice for choice in ValuedGraph.successors(self.state.value)]:
                if choice[1] not in expanded_states:
                    return False
            
            return True
            
            
    class MCTSState:
        def __init__(self, Value = None, Index = 0, Choices = []):
            self.value = Value.clone()
            self.round_index = Index
            self.choices = list(Choices) 
            
        def isTerminal(self):
            if ValuedGraph.isGoal(self.value): 
                return True
            else:
                return (self.round_index ==  (MAXROUNDNUM -1))
            
        def getReward(self):
            """compute the reward, penalty is considered"""
            
            k = ValuedGraph.K
            m = [0] * k
            l = list(self.value.value)
            penalty = 0
            
            for i in l:
                if i in range(1, k+1):
                    m[i - 1] += 1
            
            for i in range(k):
                if m[i] != 0:
                    penalty += (m[i]-1)**2 
            
            return self.value.score - penalty
            
        def getNextState(self):
            random_choice = random.choice([choice for choice in ValuedGraph.successors(self.value)])
            next_state = MCTSState(random_choice[1], self.round_index+1, self.choices+[random_choice])
            return random_choice[0],next_state
            

    def TreePolicy(node):
        
        while not node.state.isTerminal():
            if node.isAllExpanded():
                next_node = BestChild(node,True)
                if next_node:
                    node = next_node
            else:
                sub_node = expand(node)
                return sub_node
            
        return node    
            
    def BestChild(node, isExploration):
        
        best_score = -1
        best_child = None
                    
        for sub_node in node.children:
            
            if isExploration == True:
                C = 1.0 / math.sqrt(2.0)
            else:
                C = 0.0
            
            first = sub_node.Q / sub_node.n
            second = 2.0 * math.log(node.n) / sub_node.n
            score = first + C * math.sqrt(second)
            
            if score > best_score:
                best_score = score
                best_child = sub_node
        
        return best_child    
        
    def expand(node):
        tried_sub_states_value = set([sub_node.state.value for sub_node in node.children])
        new_state = node.state.getNextState()
        while new_state[1].value in tried_sub_states_value:
            new_state = node.state.getNextState()
            
        sub_node = MCTSNode(new_state[1], node, new_state[0])
        node.addChild(sub_node)
        return sub_node
        
    def RolloutPolicy(node):
        """Simulate until the goal is reached or the maximum round number is reached"""
        
        current_state = node.state
        while not current_state.isTerminal():
            current_state = current_state.getNextState()[1]

        reward = current_state.getReward()
        return reward
    
    def backup(node, reward):
        while(node != None):
            node.n += 1
            node.Q += reward
            node = node.parent
    
    
    
    state = MCTSState(state)
    node = MCTSNode(state)
    for i in range(budget):
        expand_node = TreePolicy(node)
        reward = RolloutPolicy(expand_node)
        backup(expand_node, reward)
    
    
    best_child = BestChild(node,False)
    if not best_child:
        return node.state.value
    return best_child.state.value
        
   

        
def MCTSSearch(ValuedGraph, initialState):
    """repeatedly search for the best next state"""
    path = [initialState.value]
    state = initialState
    state.printState(ValuedGraph.N, ValuedGraph.M)
    print(state.score)
    while (not ValuedGraph.isGoal(state)):
        state = (MCTS(ValuedGraph,state))
        path.append(state.value)
        state.printState(ValuedGraph.N, ValuedGraph.M)
        print(state.score)
    
    return (path,state.score)



def clickomaniaPlayer(clickomania, initialState):
    """play the game
    Notice that the initialState must use the State class given in clickomania.py
    As done in generateState function, assume s is a list with the length to be N * M, and K types of tiles
    the state should be initialized as    initialState = ck.State(s)"""
    
    return MCTSSearch(clickomania, initialState)
    
