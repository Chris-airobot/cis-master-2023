import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import functools
import random
import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv
from pettingzoo import ParallelEnv
from src.utils import *


class SingleEnvironment(ParallelEnv):
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
        self.bridges = None


        self.auxiliary = None

        # 0 for solver, 1 for helper
        self.current_agent = 0 


        
    def reset(self, seed=None, return_info=False, options=None):

        # solver coordinates
        self.solver_x = random.randint(0, 6)
        self.solver_y = random.randint(0, 6)
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



        


        ##################################################
        #### Map in hard mode ############################
        ##################################################
        # self.grid, self.bridges = hard_mode_map((0,0), (self.door_x, self.door_y))


        ######################################################
        #### Map in one path mode ############################
        ######################################################        
        self.grid, self.bridges = one_path_map(self.solver_x, self.solver_y, self.door_x, self.door_y)    
    
        ######################################################
        #### Map in comlex mode ##############################
        ######################################################        
        # self.grid, self.bridges = complex_map_generation(self.solver_x, self.solver_y, self.door_x, self.door_y) 


        self.timestep = 0   
        self.discount_factor = 0.99
        self.current_agent = 0
        self.auxiliary = random.choice(np.arange(-1,1.1,0.1))
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



        r_timeout = -self.timestep * 0.05

        # Manhattan Distance before the solver moves, potential function 
        potential_1 = 1 - (abs(self.solver_x - self.door_x) + abs(self.solver_y - self.door_y)) / 12 


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
        r_closer = 0.5 if r_closer > 0 else -0.2
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

        return Discrete(9)
