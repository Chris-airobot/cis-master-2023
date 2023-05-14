from utils import *
import copy
import numpy as np

solver_x = random.randint(0,6)
solver_y = random.randint(0,6)
door_x = random.randint(2,5)
door_y = random.randint(2,5)
grid = np.zeros((7, 7), dtype=object)




##################################################
#### Map in hard mode ############################
##################################################
# grid, bridges = hard_mode_map((0, 0), (door_x, door_y))

######################################################
#### Map in one path mode ############################
######################################################        
grid, bridges = one_path_map(solver_x, solver_y, door_x, door_y)   

for x in range(len(grid)):
    for y in range(len(grid[0])):
        if grid[x,y] == 1:
            grid[x,y] = "1"
        if grid[x,y] == 0:
            grid[x,y] = "0"
    
grid[solver_x, solver_y] = "S"
grid[door_x,door_y] = "G"    
print(grid)