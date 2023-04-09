import gym
import numpy as np
from agents import Agent
from utils import *
import random
import os
os.system('clear')

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    i = 0
    myPQ = PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, '',0, [])
    myPQ.push(startNode, heuristic(startState, problem))
    visited = set()
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, action, cost, path = node
        #print(f'Your state is {state}, action is {action}, cost is {cost}, path is {path}')

        if (not state in visited) or cost < best_g.get(state):
            visited.add(state)
            best_g[state]=cost
            #print(f'All right, your best_g is: {best_g[state]}')

            i += 1
            # print(f'For each loop, your state is {state[1]}')
            # print(f'Right now it is the {i}th loop')
            if problem.isGoalState(state):
                path = path + [(state, action)]
                actions = [action[1] for action in path]
                del actions[0]
                return actions
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                newNode = (succState, succAction, cost + succCost, path + [(node, action)])

                myPQ.push(newNode,heuristic(succState,problem)+cost+succCost)

def nextState(current):
    connected_points = []
    if current[0] > 0:
        connected_points.append([(current[0]-1, current[1]), 'go up one block'])
    if current[0] < 6:
        connected_points.append([(current[0]+1, current[1]), 'go down one block'])
    if current[1] > 0:
        connected_points.append([(current[0], current[1]-1), 'go left one block'])
    if current[1] < 6:
        connected_points.append([(current[0], current[1]+1), 'go right one block'])
    if current[0] > 1:
        connected_points.append([(current[0]-2, current[1]), 'go up two blocks'])
    if current[0] < 5:
        connected_points.append([(current[0]+2, current[1]), 'go down two blocks'])
    if current[1] > 1:
        connected_points.append([(current[0], current[1]-2), 'go left two blocks'])
    if current[1] < 5:
        connected_points.append([(current[0], current[1]+2), 'go right two blocks'])
    
    return connected_points 

# BFS to check if map exists the path 
def BFS(grid: np.array, current: tuple, visited: list, end:tuple):
    myQ = Queue()
    # start state
    visited.append(current)
    myQ.push([current,[]])
    print('are you here')
    # myQ.push(current)
    while not myQ.isEmpty():
        coord, actions = myQ.pop()
        # coord = myQ.pop()
        
        if grid[coord[0]][coord[1]] == '1' or grid[coord[0]][coord[1]] == 'G' or grid[coord[0]][coord[1]] == 'P':
            
            # print(f'coord is: {coord}')
            # print(f'end is: {end}')
            if coord == end:    
            
                # print(f"Goal is: {end}")
                # print(f"There is a path")     
                return True, actions
                # return True
            else:
                for successor in nextState(coord):
                    next_state, action = successor
                    if next_state not in visited:
                        visited.append(next_state)
                        myQ.push([next_state, actions+[action]])
                        # myQ.push(successor)

    return False, []


solver_only = [38, 40, 37, 33, 51, 31, 44, 43, 32, 38]
print(np.average(solver_only)/300)