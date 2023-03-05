import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete


from pettingzoo.utils.env import ParallelEnv

GENERATOR_MOVES = ["DISTANCE", "ANGLE", "HEIGHT", "SIZE"]
SOLVER_MOVES = ["FORWARD", "BACKWARD", "TURN", "JUMP"]


class CustomEnvironment(ParallelEnv):
    def __init__(self):
        self.goal_position = (None, None, None)        
        self.generator_position = (None, None, None)
        self.solver_position = (None, None, None)
       
        # Observation Parameters
        self.relative_position = None
        self.distance_to_goal = None
        # I think this angle simply means the xy plane angle
        self.solver_goal_angle = None 
        self.previous_block = None
        self.auxiliary = 1

        self.timestep = None
        self.possible_agents = ["solver", "generator"]
        # used for track last block
        # In the paper, they said they track last two block's angle
        # Doesn't really make sense to me 
  
 
    # generator actions
    def generate_blocks(self):
        size = random.uniform(4, 6)
        self.new = (random.uniform(-np.sqrt(10), np.sqrt(10)), 
                    random.uniform(-np.sqrt(10), np.sqrt(10)),
                    random.uniform(-2, 2)) 
        
        distance = np.sqrt(self.new[0]**2 + self.new[1]**2)

        # distance to next_generator
        while distance > 10 and distance < 5:
            self.new = (random.uniform(-np.sqrt(10), np.sqrt(10)), 
                    random.uniform(-np.sqrt(10), np.sqrt(10)),
                    random.uniform(-2, 2))
            distance = np.sqrt(self.new[0]**2 + self.new[1]**2)


        # Case for generating the block for the first time, we record the size here for next transistion
        # So the first update of the size is the current, but fututre updates of the size is the last 
        if self.previous_block == None:
            self.previous_block = [(0,0,0), size, None]
        
        else:
            # angle
            angle_to_last_block = np.arccos((self.generator_position[0] * self.previous_block[0][0] + 
                               self.generator_position[1] * self.previous_block[0][1]) 
                               / np.sqrt((self.generator_position[0]**2 + self.generator_position[1]**2)*
                                         (self.previous_block[0][0]**2 + self.previous_block[0][1]**2)))
            self.previous_block = [self.generator_position, self.previous_block[1], angle_to_last_block]
        
        # new_generator_position
        self.generator_position = tuple(map(lambda i, j: i + j, self.previous_block[0], self.new))
        
        



    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # Initialize for three elements
        self.solver_position = (0,0,0)
        self.goal_position = (random.uniform(50,55), random.uniform(50,55), random.uniform(50,55))
        #self.generate_blocks()
        

        self.relative_position = tuple(map(lambda i, j: i - j, self.solver_position, self.goal_position))

        # 3D angle
        # self.solver_goal_angle = np.arccos(
        #     np.sum(list(map(lambda i, j: i * j, self.solver_position, self.goal_position)))
        #     / np.sqrt((np.sum(i*i for i in self.solver_position))
        #     *(np.sum(i*i for i in self.goal_position))))
        
        # 2D angle
        self.solver_goal_angle = np.arccos((self.solver_position[0] * self.goal_position[0] + 
                                           self.solver_position[1] * self.goal_position[1])
                                        / np.sqrt((self.solver_position[0]**2 + self.solver_position[1]**2) * 
                                                  (self.goal_position[0]**2 + self.goal_position[1]**2)))

        self.distance_to_goal = np.sqrt(sum(list(lambda i, j: (i - j)**2, 
                                                 self.goal_position, 
                                                 self.solver_position)))
        


        observation = [
            self.relative_position,
            self.solver_goal_angle,
            self.distance_to_goal,
            None,
            self.auxiliary
            ]

        observations = {
            'solver': {'observation': observation},
            'generator':{'observation': observation},
        }


        return observations

    def step(self, actions):
        # Execute actions
        solver_action = actions['solver']
        generator_action = actions['generator']

        # E.g. solver_action = [forward_dist: 2, turn:30, jump:0.8]
        #      generator_action = [dististance_to_next_block: 6,
        #                          angle_to_next_block: 30,
        #                          square_size: 5
        #                          height_change: 1]

        if solver_action == 0 and self.solver_x > 0:
            self.solver_x -= 1
        if solver_action == 1 and self.solver_x < 6:
            self.solver_x += 1
        if solver_action == 2 and self.solver_y > 0:
            self.solver_y -= 1
        if solver_action == 3 and self.solver_y < 6:
            self.solver_y += 1    

        if generator_action == 0 and self.generator_x > 0:
            self.generator_x -= 1
        if generator_action == 1 and self.generator_x < 6:
            self.generator_x += 1
        if generator_action == 2 and self.generator_y > 0:
            self.generator_y -= 1
        if generator_action == 3 and self.generator_y < 6:
            self.generator_y += 1    



        # Termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.solver_x == self.generator_x and self.solver_y == self.generator_y:
            rewards = {'solver':-1, 'generator':1}
            terminations = {a:True for a in self.agents}
            self.agents = []

        elif self.solver_x == self.goal_x and self.solver_y == self.goal_y:
            rewards = {'solver':1, 'generator':-1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        # Truncation conditions
        truncations = {"solver": False, "generator": False}
        if self.timestep > 100:
            rewards = {'solver': 0, 'generator': 0}
            truncations = {'solver': True, 'generator': True}
            self.agents = []
        self.timestep +=1

        # Observation
        observation = {
                self.solver_x + 7 * self.solver_y,
                self.generator_x + 7 * self.generator_y,
                self.goal_x + 7 * self.goal_y,
        }

        observations = {
            'solver': {
                'observation': observation
            },
            'generator':{
                'observation': observation
            }
        }



        # Info
        infos = {"solver": {}, "generator": {} }

        return observations, rewards, terminations, truncations, infos


    def render(self):
        grid = np.zeros((7,7))
        grid[self.solver_y, self.solver_x] = 'P'
        grid[self.generator_y, self.generator_x] = 'G'
        grid[self.goal_y, self.goal_x] = 'E'
        print(f'{grid} \n')

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([7 * 7 -1] * 3)
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)
    
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    parallel_api_test(CustomEnvironment(), num_cycles=1_000_000)