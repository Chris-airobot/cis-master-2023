
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
        # print("You are resetting")
        self.grid = np.zeros((7, 7), dtype=object)
        self.agents = copy(self.possible_agents)
        self.timestep = 0


        self.prisoner_x = 0
        self.prisoner_y = 0
        self.door_x = random.randint(2, 5) 
        self.door_y = random.randint(2, 5)

        self.path = np.zeros(8, dtype=int)

        self.grid, self.visited = map_generation(np.zeros((7, 7), dtype=object),
                                                 (self.prisoner_x, self.prisoner_y),
                                                 [],
                                                 (self.door_x, self.door_y))
        
        self.bridges = self.visited[:self.visited.index((self.door_x,self.door_y))+1]

        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.door_x + 7 * self.prisoner_y,
                self.path[0],self.path[1],  
                self.path[2],self.path[3],  
                self.path[4],self.path[5],  
                self.path[6],self.path[7],                
            )
            for a in self.agents
        }
        return observations

    def step(self, actions):
        # Check terminate conditions
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {a: "" for a in self.possible_agents}
        rewards = {a: 0 for a in self.agents}
        # closer = False
        # created = True
        r_timeout = -self.timestep * 0.03

        prisoner_action = actions["prisoner"]

        previous_distance = np.sqrt((self.prisoner_x - self.door_x)** 2 +
                                    (self.prisoner_y - self.door_y)** 2)
        
        
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

        

        # elif prisoner_action == 4:
            # print("Solver did nothing")
            # pass
        after_distance = np.sqrt((self.prisoner_x - self.door_x)** 2 +
                                 (self.prisoner_y - self.door_y)** 2)

        # print(f'previous distance: {previous_distance}, after_distance: {after_distance}')
        closer = (after_distance - previous_distance)     
        if closer == 0:
            r_closer = -0.5
        elif closer < 0:
            r_closer = 0.5
        else: 
            r_closer = -0.3

        self.checkPath()

        rewards['prisoner'] = r_closer + r_timeout
        



        
        # Solver touches the trap
        if (self.prisoner_x, self.prisoner_y) not in self.bridges:
            r_ext = -1
            # r_inc = 1 if closer else 0 

            rewards["prisoner"] += r_ext + r_timeout
            terminations = {a: True for a in self.agents}
            self.agents = []
        # Solver reach the goal
        elif self.prisoner_x == self.door_x and self.prisoner_y == self.door_y:
            r_ext = 2
            # r_inc = 0.8 if closer else 0 
            print("Reaches the Goal!")
            infos['prisoner'] = "Completed"
            rewards["prisoner"] += r_ext + r_timeout
            # rewards = {"prisoner": 1, "helper": -1}
            terminations = {a: True for a in self.agents}
            self.agents = []


        # Check truncation conditions (overwrites termination conditions)
        if self.timestep > 100:
            print("Zan'nen, time out")
            rewards["prisoner"] += r_timeout
            truncations = {"prisoner": True}
            self.agents = []
        self.timestep += 1

        # Get observations
        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.door_x + 7 * self.prisoner_y,
                self.path[0],self.path[1],  
                self.path[2],self.path[3],  
                self.path[4],self.path[5],  
                self.path[6],self.path[7],  
            )
            for a in self.possible_agents
        }

        

        return observations, rewards, terminations, truncations, infos


    def checkPath(self):
        self.path[0] = 0 if self.prisoner_x > 0 and (self.prisoner_x-1, self.prisoner_y) not in self.bridges else 1
        self.path[1] = 0 if self.prisoner_x > 1 and (self.prisoner_x-2, self.prisoner_y) not in self.bridges else 1
        self.path[2] = 0 if self.prisoner_x < 6 and (self.prisoner_x+1, self.prisoner_y) not in self.bridges else 1
        self.path[3] = 0 if self.prisoner_x < 5 and (self.prisoner_x+2, self.prisoner_y) not in self.bridges else 1
        self.path[4] = 0 if self.prisoner_y > 0 and (self.prisoner_x, self.prisoner_y-1) not in self.bridges else 1
        self.path[5] = 0 if self.prisoner_y > 1 and (self.prisoner_x, self.prisoner_y-2) not in self.bridges else 1
        self.path[6] = 0 if self.prisoner_y > 6 and (self.prisoner_x, self.prisoner_y+1) not in self.bridges else 1
        self.path[7] = 0 if self.prisoner_y > 5 and (self.prisoner_x, self.prisoner_y+2) not in self.bridges else 1



    def render(self):
        # grid = np.zeros((7, 7))

        for coord in self.bridges:
            self.grid[coord[0]][coord[1]] = 1
        # self.grid[np.isin(self.grid, 'S')] = 1
        self.grid[self.prisoner_x][self.prisoner_y] = 'P'
        # grid[self.bridges_y, self.bridges_x] = "G"
        self.grid[self.door_x][self.door_y] = 'D'
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