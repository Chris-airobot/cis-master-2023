import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers



def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = CustomEnvironment(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env









class CustomEnvironment(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}
    def __init__(self, render_mode=None):
        self.escape_y = None
        self.escape_x = None
        
        # list of coordinates that the solver can go
        self.path = [(0,0)]
        # self.helper_y = None
        # self.helper_x = None
        self.prisoner_y = None
        self.prisoner_x = None
        self.timestep = None
        self.possible_agents = ["prisoner", "helper"]
        self.grid = np.zeros((7, 7))
        self.auxiliary = 1

        # self.closer = False

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.prisoner_x = 0
        self.prisoner_y = 0

        # self.helper_x = 7
        # self.helper_y = 7
 
        self.escape_x = random.randint(2, 5)
        self.escape_y = random.randint(2, 5)
        self.path.append((self.escape_x, self.escape_y))

        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                # self.path,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }
        return observations

    def step(self, actions):
        closer = False
        created = True

        # Execute actions   
        # 0 up, 1 down, 2 left, 3 right, 4 no movements
        prisoner_action = actions["prisoner"]
        helper_action = actions["helper"]

        previous_distance = np.sqrt((self.prisoner_x - self.escape_x)** 2 +
                                    (self.prisoner_y - self.escape_y)**2)

        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < 6:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < 6:
            self.prisoner_y += 1
        elif prisoner_action == 4:
            pass
        
        after_distance = np.sqrt((self.prisoner_x - self.escape_x)** 2 +
                                 (self.prisoner_y - self.escape_y)** 2)

        closer = (after_distance - previous_distance) > 0

        # 0-3 building up/down/left/rightagent
        if helper_action == 0 and self.prisoner_x > 0:
            self.grid[self.prisoner_y][self.prisoner_x-1] = 1
            self.path.append((self.prisoner_y, self.prisoner_x-1))
        elif helper_action == 1 and self.prisoner_x < 6:
            # self.helper_x += 1
            self.grid[self.prisoner_y][self.prisoner_x+1] = 1
            self.path.append((self.prisoner_y, self.prisoner_x+1))
        elif helper_action == 2 and self.prisoner_y > 0:
            # self.helper_y -= 1
            self.grid[self.prisoner_y-1][self.prisoner_x] = 1
            self.path.append((self.prisoner_y-1,self.prisoner_x))
        elif helper_action == 3 and self.prisoner_y < 6:
            self.grid[self.prisoner_y+1][self.prisoner_x] = 1
            self.path.append((self.prisoner_y+1,self.prisoner_x))
            # self.helper_y += 1
        else:
            created = False

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if created:
            r_int = -0.1
        else:
            r_int = 0


        # Solver touches the trap
        if (self.prisoner_x, self.prisoner_y) not in self.path:
            r_ext = -3
            r_inc = 0.1 if closer else 0 
            r_prisoner = r_ext*self.auxiliary + r_inc + r_int*self.auxiliary
            rewards = {"prisoner": r_ext, "helper": r_prisoner}
            terminations = {a: True for a in self.agents}
            self.agents = []
        # Solver reach the goal
        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            r_ext = 2
            r_inc = 0.1 if closer else 0 
            r_prisoner = r_ext*self.auxiliary + r_inc + r_int*self.auxiliary
            rewards = {"prisoner": r_ext, "helper": r_prisoner}
            # rewards = {"prisoner": 1, "helper": -1}
            terminations = {a: True for a in self.agents}
            self.agents = []


        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.possible_agents}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "helper": 0}
            truncations = {"prisoner": True, "helper": True}
            self.agents = []
        self.timestep += 1

        # Get observations
        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                # self.path,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.possible_agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        # grid = np.zeros((7, 7))
        self.grid[self.prisoner_y, self.prisoner_x] = 20
        # grid[self.helper_y, self.helper_x] = "G"
        self.grid[self.escape_y, self.escape_x] = 21
        print(f"{self.grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([7 * 7 - 1] * 3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)


from pettingzoo.test import parallel_api_test # noqa: E402
from pettingzoo.test import render_test
if __name__ == "__main__":
    parallel_api_test(CustomEnvironment(), num_cycles=1_000_000)
    # render_test(CustomEnvironment)