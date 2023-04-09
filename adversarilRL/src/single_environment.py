
import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv
from pettingzoo import ParallelEnv
from utils import *


class SingleEnvironment(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}
    def __init__(self, render_mode=None):
        
        self.door_x = None
        self.door_y = None
        self.prisoner_x = None
        self.prisoner_y = None
        # 0 for trap, 1 for bridge
        self.path = np.zeros(8, dtype=int)
        self.timestep = None
        self.possible_agents = ["prisoner"]
        self.grid = None

        # self.auxiliary = 1

        
    def reset(self, seed=None, return_info=False, options=None):
        # For training
        # self.grid = np.loadtxt('map.txt', dtype=object)
        # self.bridges = [list(i) for i in np.loadtxt('bridges.txt', dtype=int)]
        # self.door_x, self.door_y = random.choice([(4,2), (4, 3), (4, 4), (4, 5), (4, 6),(5, 0),(5, 3), (5, 4)])
        checked = False
        # For testing
        



        self.agents = copy(self.possible_agents)
        self.timestep = 0
        


        self.prisoner_x = 0
        self.prisoner_y = 0
        
        self.door_x = random.randint(2,5)
        self.door_y = random.randint(2,5)

             
        
        while not checked:
            self.grid, self.bridges = map_generation(self.prisoner_x, 
                                                     self.prisoner_y,
                                                     self.door_x, 
                                                     self.door_y)
            
            checked = map_check(self.grid,(self.prisoner_x, self.prisoner_y), [], (self.door_x, self.door_y))

        self.path = np.zeros(8, dtype=int)   
        self.checkPath()
        
        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.door_x + 7 * self.door_y,
                self.path[0],self.path[1],  
                self.path[2],self.path[3],  
                self.path[4],self.path[5],  
                self.path[6],self.path[7],                
            )
            for a in self.agents
        }

        return observations

    def step(self, actions):
        # Initialize all variables for return
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {a: "" for a in self.possible_agents}
        rewards = {a: 0 for a in self.agents}
    
        
        prisoner_action = actions["prisoner"]
        heuristic_before = BFS(self.grid, (self.prisoner_x, self.prisoner_y), [], (self.door_x, self.door_y))

        # Execute actions   
        # 1 block move: 0 up, 1 down, 2 left, 3 right
        # 2 block move: 4 up, 5 down, 6 left, 7 right
        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
            # print("Solver moves up")
        elif prisoner_action == 1 and self.prisoner_x < 6:
            self.prisoner_x += 1
            # print("Solver moves down")
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
            # print("Solver moves left")
        elif prisoner_action == 3 and self.prisoner_y < 6:
            self.prisoner_y += 1
            # print("Solver moves right")
        
        # 2 block move: 4 up, 5 down, 6 left, 7 right
        elif prisoner_action == 4 and self.prisoner_x > 1:
            self.prisoner_x -= 2
            # print("Solver moves up")
        elif prisoner_action == 5 and self.prisoner_x < 5:
            self.prisoner_x += 2
            # print("Solver moves down")
        elif prisoner_action == 6 and self.prisoner_y > 1:
            self.prisoner_y -= 2
            # print("Solver moves left")
        elif prisoner_action == 7 and self.prisoner_y < 5:
            self.prisoner_y += 2

        

        heuristic_after = BFS(self.grid, (self.prisoner_x, self.prisoner_y), [], (self.door_x, self.door_y))

        # If the solver moves to the trap , it gets the negatve reward
        if heuristic_after >= 0:
            r_closer = (heuristic_before - heuristic_after) / max(heuristic_after, heuristic_before) * 2 
        else:
            r_closer = -1 /heuristic_before * 2 

        # The function that fills in the "path" in observation
        self.checkPath()

        rewards['prisoner'] = r_closer if r_closer != 0 else -0.2

        # Aims to make the solver moves as fast as possible
        r_timeout = -self.timestep * 0.03



        
        # Solver touches the trap
        if [self.prisoner_x, self.prisoner_y] not in self.bridges:
            r_ext = -2
            rewards["prisoner"] += r_ext + r_timeout
            terminations = {a: True for a in self.agents}
            self.agents = []
        # Solver reach the goal
        elif self.prisoner_x == self.door_x and self.prisoner_y == self.door_y:
            r_ext = 2         
            infos['prisoner'] = "Completed"
            rewards["prisoner"] += r_ext + r_timeout
            print(f"Reaches the Goal! prisoner's reward for this step is {rewards['prisoner']}")
            terminations = {a: True for a in self.agents}
            self.agents = []


        # Check truncation conditions (overwrites termination conditions)
        if self.timestep > 20:
            rewards["prisoner"] += r_timeout
            print(f"Time out")
            infos['prisoner'] = "out"
            truncations = {"prisoner": True}
            self.agents = []
        self.timestep += 1

        # Get observations
        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.door_x + 7 * self.door_y,
                self.path[0],self.path[1],  
                self.path[2],self.path[3],  
                self.path[4],self.path[5],  
                self.path[6],self.path[7],  
            )
            for a in self.possible_agents
        }

        

        return observations, rewards, terminations, truncations, infos


    def checkPath(self):
        # up one
        self.path[0] = 1 if self.prisoner_x > 0 and [self.prisoner_x-1, self.prisoner_y] in self.bridges else 0
        # up two
        self.path[1] = 1 if self.prisoner_x > 1 and [self.prisoner_x-2, self.prisoner_y] in self.bridges else 0
        # down one
        self.path[2] = 1 if self.prisoner_x < 6 and [self.prisoner_x+1, self.prisoner_y] in self.bridges else 0
        # down two
        self.path[3] = 1 if self.prisoner_x < 5 and [self.prisoner_x+2, self.prisoner_y] in self.bridges else 0
        # left one
        self.path[4] = 1 if self.prisoner_y > 0 and [self.prisoner_x, self.prisoner_y-1] in self.bridges else 0
        # left two
        self.path[5] = 1 if self.prisoner_y > 1 and [self.prisoner_x, self.prisoner_y-2] in self.bridges else 0
        # right one
        self.path[6] = 1 if self.prisoner_y < 6 and [self.prisoner_x, self.prisoner_y+1] in self.bridges else 0
        # right two
        self.path[7] = 1 if self.prisoner_y < 5 and [self.prisoner_x, self.prisoner_y+2] in self.bridges else 0



    def render(self):
        # grid = np.zeros((7, 7))

        for coord in self.bridges:
            self.grid[coord[0]][coord[1]] = '1'
        
        self.grid[self.prisoner_x][self.prisoner_y] = 'P'
        # grid[self.bridges_y, self.bridges_x] = "G"
        self.grid[self.door_x][self.door_y] = 'D'

        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] == 0:
                    self.grid[x][y] = '0'
        print(f"{self.grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, _event=None):
        return MultiDiscrete([47,47,2,2,2,2,2,2,2,2])

    @functools.lru_cache(maxsize=None)
    def action_space(self, _event=None):
        return Discrete(8)


from pettingzoo.test import parallel_api_test # noqa: E402
from pettingzoo.test import render_test
if __name__ == "__main__":
    parallel_api_test(SingleEnvironment(), num_cycles=1_000_000)
    # render_test(DoubleEnvironment)ss