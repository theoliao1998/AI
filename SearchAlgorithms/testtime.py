#test the time of each algorithm with completely randomly generated initial states
#however, even with n = 5, it's not likely to be lucky enough to generate 5 initial states
#that can be found within half of an hour on average 

import search as s
import testgraphs as t
import random as rd
import time

def rand_state(n):
    """generate a random state of nPuzzle graph"""
    
    cnt = 1
    while cnt%2 != 0:
        cnt = 0
        state = rd.sample(list(range(n**2)), n**2)
        for i in range(1,n**2):
            for j in range(0,i):
                if state[i] != 0 and state[j]!=0:
                    cnt += 1
                                 
    state = t.State(state)
    return state
    

search_methods = [s.BFS, s.UCS, s.DFS, s.DLS, s.IDS, s.Astar, s.MCTSSearch]
name = ["BFS","UCS","DFS","DLS","IDS","A*","MCTS"]
n = 3
maxn = 21 #control the number of data to be collected
jump = [True, True, True, True, True, False, True]
#control which methods are to be ignored
#to be ignored if True

next_round = True
while next_round and n < maxn:
    
    graph = t.nPuzzleGraph(n)
    states = list(range(5))

    for j in range(5):
        states[j] = rand_state(n)
        print("state ",j,": ", states[j])
    
    for j in range(7):
        cost_time = 0
        if jump[j]:
            continue
            
        print(name[j])
        for i in range(5):
            initialState = states[i]
            graph.setInitialState(initialState)
            time_start=time.time()
            result = search_methods[j](graph,initialState)
            time_end=time.time()
            if result:
                print('State ', i,' time cost:',time_end-time_start,'s')
            else: 
                print("Fail")
            
            if (time_end-time_start) > 3600 or (not result):
                jump[j] = True  #once some approach costs too much time, it will not be tested anymore
                break
            else:
                cost_time += time_end-time_start
    
        if not jump[j]:
            cost_time /= 5
            print("Average time cost: ", cost_time,'s')
            if cost_time > 1800:
                jump[j] = True
        else:
            print("NA")
    
    next_round = False #if all approaches are to be ignored, terminate the program
    for j in jump:
        if not j:
            next_round = True
    n += 2
