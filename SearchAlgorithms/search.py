#search algorithms common implementation
#every graph to be search are expected to return a list of tuples
#the first element of the tuple represents the path cost to a state,
#it should be 1 for non-valued graphs
#and the second element of the tuple represents the state 
#So testgraphs is somehow modefied 

import sys
import math
import time
import random
import testgraphs as t

sys.setrecursionlimit(300000)  # set the maximum recursion depth

class Node:
    """Node for graph search"""
    
    def __init__(self, State, Parent = 0, Path_Cost = 1):
        self.state = State
        self.parent = Parent
              
        if self.parent==0:  #Parent is usually also a Node, except that self is the initial Node
            self.path_cost = 0
            self.depth = 0
        else:
            self.path_cost = Path_Cost + Parent.path_cost
            self.depth = 1 + Parent.depth
      
    def path(self):
        """return the path as a list"""
        
        if self.parent == 0:
            return [self.state]
        else:
            return self.parent.path()+[self.state]
    
class LimitedSet(set):
    def append(self, element):
        if len(self) > 50000:
            for i in range(1000):
                self.pop()
        
        self.add(element)        
        
def graph_search(graph, initialState, frontier):
    """Search through the successors of a problem to find a goal.
    The  argument frontier should be an empty queue
    graph_search doesn't explore the explored states"""
    
    explored = LimitedSet()
    frontier.append(Node(initialState))
    start = time.time()
    while frontier:
        test = time.time()

        if (test -start) > 3600: #if time cost is more than 1 hour than fail
            return None
            
        node = frontier.pop()
        if node == None:
            return None
            
        if graph.isGoal(node.state):
            return node
        
        if node.state not in explored:
            explored.append(node.state)
            extend(frontier,explored,graph,node)  
                 
    return None
    

def extend(frontier, explored, Graph, n):
    """extend the frontier"""
    
    frontier_states = []
    for node in frontier:
        frontier_states.append(node.state)
    
    succs = Graph.successors(n.state)
    #random.shuffle(succs)
    for ele in succs:
        if isinstance(ele,tuple):
            cost = ele[0]
            ele = ele[1]
        else:
            cost = 1
        
        if (ele not in explored) and (ele not in frontier_states):              
            new_node = Node(ele, n, cost)
            frontier.append(new_node)                            



def BFS(Graph, initialState):
    """breadth-first search"""

    class FIFO_Queue(list):
        """first-in-first out queue"""
        _sentinel = object()

        def pop(self):
            if self:
                result = self[0]
                del self[0]
                return result
            
    result = graph_search(Graph, initialState, FIFO_Queue())
    if result != None:    
        return(result.path(), result.path_cost)


def UCS(ValuedGraph, initialState):
    """uniform-cost search"""

    class Queue_based_on_cost(list):
        """This queue only pops the element with the least cost"""
        
        def append(self,elem):
            for node in self:
                if elem.path_cost<node.path_cost:
                    self.insert(self.index(node),elem)
                    return
                    
            self.insert(len(self), elem) 

        def pop(self):
            if self:
                result = self[0]
                del    self[0]
                return result
            

    result = graph_search(ValuedGraph, initialState, Queue_based_on_cost())
    if result != None:
        return(result.path(), result.path_cost)


def DFS(Graph, initialState):
    """depth-first search"""

    class Stack(list):
        """last-in-first out queue"""
        
        def pop(self):
            length = len(self)
            if length > 0:
                result = self[length-1]
                del    self[length-1]            
                return result

    result = graph_search(Graph, initialState, Stack())
    if result != None:
        return(result.path(), result.path_cost)

def DLS(Graph, initialState, depthLimit = 20):
    """Depth-limited search"""

    class depthLimitedStack(list):
        """stack with limitation on depth"""
        
        def append(self,elem):
            if elem.depth <= depthLimit:
                self.insert(len(self), elem)

        def pop(self):
            length = len(self)
            if length > 0:
                result = self[length-1]
                del    self[length-1]
                return result

    result = graph_search(Graph, initialState, depthLimitedStack())
    if result != None:
        return(result.path(), result.path_cost)

def IDS(Graph, initialState):
    """iterative-deepening search"""
    start = time.time()
    for depth in range(sys.maxsize):
        test =time.time()
        if (test-start)>3600:
            return None
                       
        result = DLS(Graph, initialState, depth)
        if result:
            return result


def h(node):
    """the heuristic function defined for nPuzzle graph
    Manhatten distance is used"""
    
    n = len(node.state.value)
    n = int(math.sqrt(n))
    goal = list(range(1, n**2 +1))
    goal[n**2-1] = 0
    state = list(node.state.value)
    result = 0
    for i in range(n**2):
        index1 = goal.index(i)
        index2 = state.index(i)
        a = index1 % n
        b = index2 % n
        c = (index1 - a) / n
        d = (index2 - b) / n
        result += abs(a-b) + abs(c-d)
    
    return result    

def Astar(ValuedGraph, initialState, heuristic = h):
    """Astar search"""
    
    def f(n):
        return (n.path_cost + heuristic(n))

    class PriorityQueue(list):
        
        def append(self,elem):
            length = len(self)
            for node in self:
                if f(elem)<f(node):
                    self.insert(self.index(node), elem)
                    break
            if length==len(self):
                self.insert(length,elem)

        def pop(self):
            if self:
                result = self[0]
                del    self[0]
                return result
    
    result = graph_search(ValuedGraph, initialState, PriorityQueue())
    if result != None:
        return(result.path(), result.path_cost)
    

            
def MCTS(ValuedGraph, state, budget = 100):
    """Monte Carlo Tree Search which returns a selected state
    budget reprensents the number of simulations"""
    
    MAXROUNDNUM = 1000; 
    #every simulation has at most MAXROUNDNUM rounds
    #it can only terminate in advance if the goal is reached
    
    class MCTSNode(Node):
        def __init__(self, State = None, Parent = None, Path_Cost = 1):
            self.parent = Parent
            self.children = []
            self.n = 0
            self.Q = 0.0
            self.state = State
            if self.parent==None:
                self.path_cost = 0
            else:
                self.path_cost = Path_Cost + Parent.path_cost
        
        def addChild(self, node):
            self.children += [node]
            
        def isAllExpanded(self):
            expanded_states = set([ele.state.value for ele in self.children])
            for choice in [choice for choice in ValuedGraph.successors(self.state.value)]:
                if choice[1] not in expanded_states:
                    return False
            
            return True
            
            
    class MCTSState:
        def __init__(self, Value = None, Index = 0, Choices = []):
            self.value = Value.clone()
            self.round_index = Index
            self.choices = Choices 
            
        def isTerminal(self):   
            if ValuedGraph.isGoal(self.value): 
                return True
            return (self.round_index ==  (MAXROUNDNUM -1))
            
        def getReward(self):
            """The reward computing function for nPuzzle Graph and normal search
            This can be overwritten for other games"""
            
            if ValuedGraph.isGoal(self.value): return 1
            else: return 0
            
        def getNextState(self):
            random_choice = random.choice([choice for choice in ValuedGraph.successors(self.value)])
            next_state = MCTSState(random_choice[1], self.round_index+1, self.choices+[random_choice])
            return random_choice[0],next_state
            

    def TreePolicy(node):
        
        while not node.state.isTerminal():
            if node.isAllExpanded():
                node = BestChild(node,True)
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
    """constantlt use MCTS to get next state until the goal is reached"""
    start = time.time()
    state = initialState
    path = [initialState]
    cost = 0
    while (not ValuedGraph.isGoal(state)):
        test = time.time()
        if (test-start)>=3600:
            return None        
        next_state = (MCTS(ValuedGraph,state))
        path.append(next_state)
        for ele in ValuedGraph.successors(state):
            if isinstance(ele,tuple):
                if ele[1] == next_state:
                    cost += ele[0]
                    break
            
            else:
                cost += 1
        state = next_state
            
    return (path, cost)

