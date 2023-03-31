import numpy as np
import matplotlib.pyplot as plt






prisoner_action_map = {
    0 : "Prisoner moves up 1 block", 
    1 : "Prisoner moves down 1 block", 
    2 : "Prisoner moves left 1 block", 
    3 : "Prisoner moves right 1 block", 
    4 : "Prisoner moves up 2 blocks", 
    5 : "Prisoner moves down 2 blocks", 
    6 : "Prisoner moves left 2 blocks", 
    7 : "Prisoner moves right 2 blocks", 
} 

helper_action_map = {
    0 : "Helper builds up 2 blocks", 
    1 : "Helper builds down 2 blocks", 
    2 : "Helper builds left 2 blocks", 
    3 : "Helper builds right 2 blocks", 
    4 : "Helper builds up 1 block away", 
    5 : "Helper builds down 1 block away", 
    6 : "Helper builds left 1 block away", 
    7 : "Helper builds right 1 block away", 
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



def map_generation(grid, current, visited, end):

    myQ = Queue()
    visited.append(current)
    myQ.push(current)
    while not myQ.isEmpty():
        coord = myQ.pop()
        grid[coord[0]][coord[1]] = 1
        if coord == end:
            # print(f"Goal is: {end}")
            # print(f"{grid} \n")     
            return grid, visited
        else:
            for successor in getSuccessors(coord):
                if successor not in visited:
                    visited.append(successor)
                    myQ.push(successor)


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