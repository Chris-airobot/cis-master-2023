import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv
from pettingzoo import ParallelEnv
from src.utils import *



class AlternatingEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}
    def __init__(self, render_mode=None):
        # goal coordinates
        self.door_x = None
        self.door_y = None
        # solver coordinates
        self.solver_x = None
        self.solver_y = None
        # helper coordinates
        self.helper_x = None
        self.helper_y = None
        # helper latest generated block coordinates 
        self.generated_block_x = None
        self.generated_block_y = None
        # 0 for trap, 1 for bridge
        self.path = np.zeros(8, dtype=int)
        self.timestep = None
        self.grid = None
        self.discount_factor = 0.99


        self.auxiliary = -1

        # 0 for solver, 1 for helper
        self.current_agent = 0 


        
    def reset(self, seed=None, return_info=False, options=None):

        # solver coordinates
        self.solver_x = 0
        self.solver_y = 0
        # helper coordinates
        self.helper_x = 0
        self.helper_y = 0
        # helper latest generated block coordinates 
        self.generated_block_x = 0
        self.generated_block_y = 0
        # goal coordinates
        self.door_x = random.randint(2, 5) 
        # self.door_x = random.choice([2,4,6])
        self.door_y = random.randint(2, 5)

        
        self.grid = np.zeros((7, 7), dtype=object)
        # a list contains all the moveable blocks
        self.bridges = [[0,0]]
        # "bridges" are all indexs of the brdges in the map
        self.bridges.append([self.door_x, self.door_y])
        # Setting initial bridges for the grid
        self.grid[0][0] = 1
        self.grid[self.door_x][self.door_y] = 1
        
        # self.grid, self.bridges = map_generation(self.solver_x, 
        #                                          self.solver_y, 
        #                                          self.door_x, 
        #                                          self.door_y)

        self.timestep = 0
        self.discount_factor = 0.99
        self.current_agent = 0
        
        # for render function
        self.display = np.zeros((7, 7), dtype=object)
        

        # "path" contains the condition for blocks that are 
        #  all possible for agent to move 
        self.path = np.zeros(8, dtype=int)
        self.checkPath()

        observations = [self.solver_x + 7 * self.solver_y,
                        self.door_x + 7 * self.solver_y,
                        self.helper_x + 7 * self.helper_y,
                        self.path[0],self.path[1],  
                        self.path[2],self.path[3],  
                        self.path[4],self.path[5],  
                        self.path[6],self.path[7]]
        
        return observations


    def step(self, action):
        # Initialize all variables for return
        termination = False
        truncation = False
        reward = 0
        infos = {}

        
        # self.helper_x = self.solver_x
        # self.helper_y = self.solver_y


        r_timeout = -self.timestep * 0.05

        # Manhattan Distance before the solver moves, potential function 
        potential_1 = 1 - (abs(self.solver_x - self.door_x) + abs(self.solver_y - self.door_y)) / 12 


        generated = 0
        # Helper execute actions   
        # 0-3 building up/down/left/right two blocks in a row
        # 4-7 building up/down/left/right one block
        # 8-11 jump to build the block i.e. P 0 0 -> P 0 1
        if self.current_agent == 1:
            # Helper builds up 2 blocks
            if action == 0 and self.helper_x > 1 and self.grid[self.helper_x-1][self.helper_y] == 0 and self.grid[self.helper_x-2][self.helper_y] == 0:
                self.grid[self.helper_x-1][self.helper_y] = 1
                self.grid[self.helper_x-2][self.helper_y] = 1
                self.bridges.extend(([self.helper_x-1, self.helper_y],
                                    [self.helper_x-2, self.helper_y]))
                
                self.helper_x -= 2
                generated = 2
                infos['helper'] = 'and it works!'
            # Helper builds down 2 blocks
            elif action == 1 and self.helper_x < 5 and self.grid[self.helper_x+1][self.helper_y] == 0 and self.grid[self.helper_x+2][self.helper_y] == 0:                
                self.grid[self.helper_x+1][self.helper_y] = 1
                self.grid[self.helper_x+2][self.helper_y] = 1
                self.bridges.extend(([self.helper_x+1, self.helper_y],
                                    [self.helper_x+2, self.helper_y]))
                self.helper_x += 2
                generated = 2
                infos['helper'] = 'and it works!'
            # Helper builds left 2 blocks
            elif action == 2 and self.helper_y > 1 and self.grid[self.helper_x][self.helper_y-1] == 0 and self.grid[self.helper_x][self.helper_y-2] == 0:                
                self.grid[self.helper_x][self.helper_y-1] = 1
                self.grid[self.helper_x][self.helper_y-2] = 1
                self.bridges.extend(([self.helper_x, self.helper_y-1],
                                    [self.helper_x, self.helper_y-2]))
                self.helper_y -= 2
                generated = 2
                infos['helper'] = 'and it works!'
            # Helper builds right 2 blocks
            elif action == 3 and self.helper_y < 5 and self.grid[self.helper_x][self.helper_y+1] == 0 and self.grid[self.helper_x][self.helper_y+2] == 0:
                self.grid[self.helper_x][self.helper_y+1] = 1
                self.grid[self.helper_x][self.helper_y+2] = 1
                self.bridges.extend(([self.helper_x,self.helper_y+1],
                                    [self.helper_x,self.helper_y+2]))
                self.helper_y += 2
                generated = 2
                infos['helper'] = 'and it works!'
            # Helper jumps up to build 1 block
            elif action == 4 and self.helper_x > 1 and self.grid[self.helper_x-2][self.helper_y] == 0:
                self.grid[self.helper_x-2][self.helper_y] = 1
                self.bridges.append([self.helper_x-2, self.helper_y])
                self.helper_x -= 2
                generated = 1
                infos['helper'] = 'and it works!'
            # Helper jumps down to build 1 block
            elif action == 5 and self.helper_x < 5 and self.grid[self.helper_x+2][self.helper_y] == 0:
                self.grid[self.helper_x+2][self.helper_y] = 1
                self.bridges.append([self.helper_x+2, self.helper_y])
                self.helper_x += 2
                generated = 1
                infos['helper'] = 'and it works!'
            # Helper jumps left to build 1 block
            elif action == 6 and self.helper_y > 1 and self.grid[self.helper_x][self.helper_y-2] == 0:
                self.grid[self.helper_x][self.helper_y-2] = 1
                self.bridges.append([self.helper_x, self.helper_y-2])
                self.helper_y -= 2
                generated = 1
                infos['helper'] = 'and it works!'
            # Helper jumps right to build 1 block
            elif action == 7 and self.helper_y < 5 and self.grid[self.helper_x][self.helper_y+2] == 0:
                self.grid[self.helper_x][self.helper_y+2] = 1
                self.bridges.append([self.helper_x, self.helper_y+2])
                self.helper_y += 2
                generated = 1
                infos['helper'] = 'and it works!'
            # Helper builds up 1 block
            elif action == 8 and self.helper_x > 0 and self.grid[self.helper_x-1][self.helper_y] == 0:
                self.grid[self.helper_x-1][self.helper_y] = 1
                self.bridges.append([self.helper_x-1, self.helper_y])
                self.helper_x -= 1
                generated = 1
                infos['helper'] = 'and it works!'
            # Helper builds down 1 block
            elif action == 9 and self.helper_x < 6 and self.grid[self.helper_x+1][self.helper_y] == 0:
                self.grid[self.helper_x+1][self.helper_y] = 1
                self.bridges.append([self.helper_x+1, self.helper_y])
                self.helper_x += 1
                generated = 1
                infos['helper'] = 'and it works!'
            # Helper builds left 1 block
            elif action == 10 and self.helper_y > 0 and self.grid[self.helper_x][self.helper_y-1] == 0:
                self.grid[self.helper_x][self.helper_y-1] = 1
                self.bridges.append([self.helper_x, self.helper_y-1])
                self.helper_y -= 1
                generated = 1
                infos['helper'] = 'and it works!'
            # Helper builds right 1 block
            elif action == 11 and self.helper_y < 6 and self.grid[self.helper_x][self.helper_y+1] == 0:
                self.grid[self.helper_x][self.helper_y+1] = 1
                self.bridges.append([self.helper_x, self.helper_y+1])
                self.helper_y += 1
                generated = 1
                infos['helper'] = 'and it works!'
            elif action == 12:
                infos['helper'] = 'and it works!'
            else:
                infos['helper'] = 'but it does not work'


            # potential for helper
            potential_3 = 1 - (abs(self.helper_x - self.door_x) + abs(self.helper_y - self.door_y)) / 12 

            # f = open('src/saved_files/solver_status.txt', 'r')
            # if f.mode=='r':
            #     contents= f.read()
            # r_closer = float(contents)


             # Generator's normal reward's part 1, solver gets closer
            # r_inc = r_closer if r_closer > 0 else 0

            # Generator's normal reward's part 2, generator created blocks
            # r_internal = 0.2 * generated  
            r_internal = generated
            r_penalty = -2 if r_internal == 0 else 0 
            
            r_to_goal = (self.discount_factor * potential_3 - potential_1) * 4
            
            # print(f'r_internal is: {r_internal}')

            # Original trained well rewards
            # print(f'r_interal is: {r_internal * self.auxiliary}')
            # # print(f'r_inc is: {r_inc}')
            # print(f'r_to_goal is: {r_to_goal}')
            # print(f'r_timeout is: {r_timeout}')

            reward = r_internal * self.auxiliary  + r_penalty + r_to_goal
            # print(f'final reward is: {reward}')

        # Prionser execute actions   
        # 1 block move: 0 up, 1 down, 2 left, 3 right
        # 2 block move: 4 up, 5 down, 6 left, 7 right
        elif self.current_agent == 0:
            if action == 0 and self.solver_x > 0:
                self.solver_x -= 1
                infos['solver'] = 'and it works!'
                # print("Solver moves up")
            elif action == 1 and self.solver_x < 6:
                self.solver_x += 1
                infos['solver'] = 'and it works!'
                # print("Solver moves down")
            elif action == 2 and self.solver_y > 0:
                self.solver_y -= 1
                infos['solver'] = 'and it works!'
                # print("Solver moves left")
            elif action == 3 and self.solver_y < 6:
                self.solver_y += 1
                infos['solver'] = 'and it works!'
                # print("Solver moves right")
            
            # 2 block move: 4 up, 5 down, 6 left, 7 right
            elif action == 4 and self.solver_x > 1:
                self.solver_x -= 2
                infos['solver'] = 'and it works!'
                # print("Solver moves up")
            elif action == 5 and self.solver_x < 5:
                self.solver_x += 2
                infos['solver'] = 'and it works!'
                # print("Solver moves down")
            elif action == 6 and self.solver_y > 1:
                self.solver_y -= 2
                infos['solver'] = 'and it works!'
                # print("Solver moves left")
            elif action == 7 and self.solver_y < 5:
                self.solver_y += 2
                infos['solver'] = 'and it works!'
            elif action == 8:
                infos['solver'] = 'and it works!'
                pass
            else:
                infos['solver'] = 'but it does not work!'


            # Manhattan Distance after the solver moves, potential function after the movement
            potential_2 = 1 - (abs(self.solver_x - self.door_x) + abs(self.solver_y - self.door_y)) / 12


            # Solver's normal reward
            r_closer = (self.discount_factor * potential_2  - potential_1) * 6  

            # Save the best score for next ietration of training 
            # file = open("src/saved_files/solver_status.txt", "w")
            # #convert variable to string
            # str = repr(r_closer)
            # file.write(str)

            reward = r_closer + r_timeout



        self.checkPath()
        self.timestep += 1

        # Solver touches the trap
        if [self.solver_x, self.solver_y] not in self.bridges:
            r_fail = -2
            reward += r_fail if self.current_agent == 0 else r_fail * self.auxiliary
            termination = True
            infos['solver'] += ' But it fails.'
        # Solver reach the goal
        elif self.solver_x == self.door_x and self.solver_y == self.door_y:
            r_complete = 4
            reward += r_complete if self.current_agent == 0 else 0 
            infos['solver'] = "Completed"
            print("Reaches the Goal!")
            termination = True

        # Check truncation conditions (overwrites termination conditions)
        if self.timestep > 21:
            # print(f'Current agent: {self.current_agent}')
            reward += r_timeout * 4
            print("Time out")
            truncation = True

        

        # Get observations
        observation = [self.solver_x + 7 * self.solver_y,
                       self.door_x + 7 * self.solver_y,
                       self.helper_x + 7 * self.helper_y,
                       self.path[0],self.path[1],  
                       self.path[2],self.path[3],  
                       self.path[4],self.path[5],  
                       self.path[6],self.path[7]]

        

        return observation, reward, termination, truncation, infos


    def checkPath(self):
        # up one
        self.path[0] = 1 if self.solver_x > 0 and [self.solver_x-1, self.solver_y] in self.bridges else 0
        # up two
        self.path[1] = 1 if self.solver_x > 1 and [self.solver_x-2, self.solver_y] in self.bridges else 0
        # down one
        self.path[2] = 1 if self.solver_x < 6 and [self.solver_x+1, self.solver_y] in self.bridges else 0
        # down two
        self.path[3] = 1 if self.solver_x < 5 and [self.solver_x+2, self.solver_y] in self.bridges else 0
        # left one
        self.path[4] = 1 if self.solver_y > 0 and [self.solver_x, self.solver_y-1] in self.bridges else 0
        # left two
        self.path[5] = 1 if self.solver_y > 1 and [self.solver_x, self.solver_y-2] in self.bridges else 0
        # right one
        self.path[6] = 1 if self.solver_y < 6 and [self.solver_x, self.solver_y+1] in self.bridges else 0
        # right two
        self.path[7] = 1 if self.solver_y < 5 and [self.solver_x, self.solver_y+2] in self.bridges else 0



    def render(self):
        # grid = np.zeros((7, 7))

        for coord in self.bridges:
            self.display[coord[0]][coord[1]] = '1'        
        self.display[self.solver_x][self.solver_y] = 'S'
        self.display[self.door_x][self.door_y] = 'G'

        for x in range(len(self.display)):
            for y in range(len(self.display[x])):
                if self.display[x][y] == 0:
                    self.display[x][y] = '0'

        print(f"{self.display} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, _event=None):
        return MultiDiscrete([47,47,47,2,2,2,2,2,2,2,2])

    @functools.lru_cache(maxsize=None)
    def action_space(self, _event=None):
        if self.current_agent == 0:
            return Discrete(9)
        else:
            return Discrete(13)
    



 