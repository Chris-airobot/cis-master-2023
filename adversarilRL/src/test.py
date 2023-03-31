import numpy as np
import random
import time

# 1. randomly generate a grid, all elements are 0
# 2. mark the door and the prisoner's coordinates as 1
# 3. use DFS to generate a path between door and prisoner

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


prisoner_x = 0
prisoner_y = 0
door_x = random.randint(2,5)
door_y = random.randint(2,5)

grid = np.zeros((7, 7), dtype=object)
# grid[prisoner_x][prisoner_y] = grid[door_x][door_y] = 1
# visited = [(prisoner_x,prisoner_y), (door_x, door_y)]
visited = []


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
    
    return connected_points 



def BFS(grid, current, visited, end):

    myQ = Queue()
    visited.append(current)
    myQ.push(current)
    while not myQ.isEmpty():
        # time.sleep(1)
        # print(f"Goal is: {end}")
        # print(f"{grid} \n")  
        coord = myQ.pop()
        grid[coord[0]][coord[1]] = 1
        if coord == end:
            
            print(f"Goal is: {end}")
            print(f"{grid} \n")     
            return visited
        else:
            for successor in getSuccessors(coord):
                if successor not in visited:
                    visited.append(successor)
                    myQ.push(successor)
                    



def DFS(grid, current, visited, end):
    
    grid[current[0]][current[1]] = 1
    print(f"Goal is: {end}")
    print(f"{grid} \n")
    time.sleep(1)
    if grid[end[0]][end[1]] == 1:
        return

    visited.append(current)
    # up/down/left/right
    connected_points = []
    if current[0] > 0:
        connected_points.append((current[0]-1, current[1]))
    if current[0] < 6:
        connected_points.append((current[0]+1, current[1]))
    if current[1] > 0:
        connected_points.append((current[0], current[1]-1))
    if current[1] < 6:
        connected_points.append((current[0], current[1]+1))
    for point in connected_points:
        if point not in visited:
            DFS(grid, point, visited, end)


