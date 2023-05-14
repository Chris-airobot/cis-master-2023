import numpy as np
import matplotlib.pyplot as plt
import random
import heapq
import copy

REWARD_INDEX = 4
DONE_INDEX = 5
TRUNCATED_INDEX = 6


solver_action_map = {
    0 : "Prisoner moves up 1 block", 
    1 : "Prisoner moves down 1 block", 
    2 : "Prisoner moves left 1 block", 
    3 : "Prisoner moves right 1 block", 
    4 : "Prisoner jumps up 2 blocks", 
    5 : "Prisoner jumps down 2 blocks", 
    6 : "Prisoner jumps left 2 blocks", 
    7 : "Prisoner jumps right 2 blocks", 
    8 : "Prisoner does not move"
} 

helper_action_map = {
    0 : "Helper builds up 2 blocks", 
    1 : "Helper builds down 2 blocks", 
    2 : "Helper builds left 2 blocks", 
    3 : "Helper builds right 2 blocks", 
    4 : "Helper jumps up 1 block", 
    5 : "Helper jumps down 1 block", 
    6 : "Helper jumps left 1 block", 
    7 : "Helper jumps right 1 block", 
    8 : "Helper builds up 1 block", 
    9 : "Helper builds down 1 block", 
    10 : "Helper builds left 1 block", 
    11 : "Helper builds right 1 block", 
    12 : "Helper does not build"
} 


class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0


def plot_learning_curve(x, scores, figure_file, name):
    f1 = plt.figure()
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title(f'Running average of previous 100 {name} scores')
    plt.savefig(figure_file)


# BFS to check if map exists the path 
def map_check(grid: np.array, current: tuple, visited: list, end:tuple):
    myQ = Queue()
    visited.append(current)
    myQ.push(current)
    while not myQ.isEmpty():
        coord = myQ.pop()
        if grid[coord[0]][coord[1]] == 1 or grid[coord[0]][coord[1]] == 'G' or grid[coord[0]][coord[1]] == 'P':
            if coord == end:     
                return True
            else:
                for successor in getSuccessors(coord):
                    if successor not in visited:
                        visited.append(successor)
                        myQ.push(successor)
        
    # print("No valid path")  
    return False    


# BFS to check if map exists the path 
def BFS(grid: np.array, current: tuple, visited: list, end:tuple):
    myQ = Queue()
    visited.append(current)
    myQ.push([current,[]])
    while not myQ.isEmpty():
        coord, actions = myQ.pop()
        
       
        if grid[coord[0]][coord[1]] == '1' or grid[coord[0]][coord[1]] == 'D' or grid[coord[0]][coord[1]] == 'P':
            if coord == end:    
                return len(actions)
            else:
                for successor in nextState(coord):
                    
                    next_state, action = successor
                    if next_state not in visited:
                        visited.append(next_state)
                        myQ.push([next_state, actions+[action]])

    return -1


def complex_map_generation(prisoner_x, prisoner_y, door_x, door_y):
    bridges = []
    grid = np.zeros((7, 7), dtype=object)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            grid[i][j] = random.randint(0,1)
            if grid[i][j] == 1:
                bridges.append([i,j])

    grid[prisoner_x][prisoner_y] = 1
    grid[door_x][door_y] = 1

    if (prisoner_x,prisoner_y) not in grid:
        bridges.append([prisoner_x,prisoner_y])
    if (door_x,door_y) not in grid:
        bridges.append([door_x,door_y])
    return grid, bridges


def path_find(grid: np.array, current: tuple,  end:tuple):
    parent = {}
    visited = []
    myQ = Queue()

    visited.append(current)
    myQ.push(current)
    # count = 0
    while not myQ.isEmpty():
        coord = myQ.pop()
        if grid[coord[0]][coord[1]] == 1 or grid[coord[0]][coord[1]] == 'G' or grid[coord[0]][coord[1]] == 'P':

            if coord == end: 
                path = []   
                while coord in parent:
                    path.append([coord[0], coord[1]])
                    coord = parent[coord]
                path.append([current[0],current[1]])
                path.reverse()
                return True, path
            else:
                for successor in getSuccessors(coord):
                    if successor not in visited:
                        parent[successor] = coord
                        visited.append(successor)
                        myQ.push(successor)
            # count += 1
            # if count == 6:
            #     quit()
    # print("No valid path")  
    return False, []  



def hard_mode_map(current: tuple, end:tuple):
    grid = np.zeros((7, 7), dtype=object)

    for i in range(0, end[1]+1):
        grid[end[0], i] = 1

    for j in range(0, end[0]):
        grid[j, 0] = 1

    checked = False
    while not checked:
        test = copy.deepcopy(grid)
        indices = np.argwhere(test == 1)
        np.random.shuffle(indices)
        num_to_change = random.randint(1, len(indices)//2)
        # num_to_change = 4
        # print(num_to_change)
        selected_indices = indices[:num_to_change]
        # print(selected_indices)
        test[selected_indices[:, 0], selected_indices[:, 1]] = 0
        test[current[0],current[1]] = 1
        checked, bridges = path_find(test, current,  end)

    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if [x,y] not in bridges:
                grid[x,y] = 0
    return grid, bridges  


def one_path_map(solver_x, solver_y, door_x, door_y):
    checked = False
    while not checked:
        grid, bridges = complex_map_generation(solver_x, solver_y, door_x, door_y)
        checked, path = path_find(grid, (solver_x, solver_y), (door_x, door_y))

        grid = np.zeros((7, 7), dtype=object)
        for x,y in path:
            grid[x,y] = 1

    return grid, path











def getSuccessors(current):
    connected_points = []
    if current[0] > 0:
        connected_points.append((current[0]-1, current[1]))
    if current[0] < 6:
        connected_points.append((current[0]+1, current[1]))
    if current[1] > 0:
        connected_points.append((current[0], current[1]-1))
    if current[1] < 6:
        connected_points.append((current[0], current[1]+1))
    if current[0] > 1:
        connected_points.append((current[0]-2, current[1]))
    if current[0] < 5:
        connected_points.append((current[0]+2, current[1]))
    if current[1] > 1:
        connected_points.append((current[0], current[1]-2))
    if current[1] < 5:
        connected_points.append((current[0], current[1]+2))
    
    return connected_points 







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




















class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)