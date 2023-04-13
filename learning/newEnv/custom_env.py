import gym
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
from gym import spaces
from gym.envs.registration import EnvSpec
import copy
import time

from gym.spaces import flatten_space
from PPO import PPO

def flatten_action(action, space):
    if isinstance(space, gym.spaces.Dict):
        flat = {}
        for key, val in space.spaces.items():
            flat.update(flatten_action(action[key], val))
        return flat
    else:
        return {space.name: action}
    
    
class JumpingGame(AECEnv):
    def __init__(self):
        self.metadata = {'render.modes': ['human']}
        self.agents = ["solver", "generator"]
        self.agent_name_mapping = dict(zip(list(range(2)), self.agents))
        self.action_space = spaces.Box(low=np.array([0, 0, 15, 15, 80]), high=np.array([30, 30, 30, 30, 300]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.reward = {agent: 0 for agent in self.agents}
        self.done = {agent: False for agent in self.agents}
        self.generator_last_x = 0
        self.generator_last_y = 0
        self.platforms = []
        self.goal_x = 0
        self.goal_y = 0
        self.auxiliary = 0
        self.spec = EnvSpec("JumpingGame-v0", max_episode_steps=1000, reward_threshold=100.0)
        self.render_mode = 'human'
    

    def reset(self):
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.generator_last_x = 0
        self.generator_last_y = 0
        self.platforms = []
        self.goal_x = np.random.randint(0, 500)
        self.goal_y = np.random.randint(200, 400)
        self.auxiliary = np.random.uniform(-1, 1)

        return {agent: np.zeros((6,), dtype=np.float32) for agent in self.agents}

    def observe(self, agent):
        solver_x, solver_y = self.platforms[0]
        return np.array([solver_x - self.goal_x, solver_y - self.goal_y, self.generator_last_x, self.generator_last_y, self.platforms[-1][0]-self.platforms[-2][0], self.auxiliary])

    def step(self, action):
        agent = self.agent_name_mapping[agent_selector(self.dones, agent)]
        if agent == "solver":
            reward = 0
            platform_width = self.platforms[-1][0] - self.platforms[-2][0]
            new_x = self.platforms[0][0] + action[1]
            new_y = self.platforms[0][1] + action[0]
            if new_x < 0:
                new_x = 0
            elif new_x > 500:
                new_x = 500
            if new_y < 0:
                new_y = 0
            elif new_y > 500:
                new_y = 500
            self.platforms[0] = (new_x, new_y)
            if new_y <= self.platforms[1][1] and new_x >= self.platforms[1][0] and new_x <= self.platforms[1][0]+platform_width:
                reward += 1
            elif new_y > self.platforms[1][1]:
                reward -= 1
            elif new_x < self.platforms[1][0] or new_x > self.platforms[1][0]+platform_width:
                reward -= 0.5
            self.rewards[agent] = reward

        else:  # agent is "generator"
            new_platform_x = self.generator_last_x + action[3]
            new_platform_y = self.generator_last_y + action[2]
            new_platform_width = action[4]
            if new_platform_x < 0:
                new_platform_x = 0
            elif new_platform_x + new_platform_width > 500:
                new_platform_x = 500 - new_platform_width
            if new_platform_y < 0:
                new_platform_y = 0
            elif new_platform_y > 500:
                new_platform_y = 500
            self.platforms.append((new_platform_x, new_platform_y))
            self.platforms.append((new_platform_x + new_platform_width, new_platform_y))
            self.generator_last_x = new_platform_x + new_platform_width
            self.generator_last_y = new_platform_y
            generator_reward_int = 0.5 * ((new_platform_x + new_platform_width) - self.platforms[-2][0])
            generator_reward_ext = self.rewards["solver"] * 0.5
            reward = generator_reward_int * self.auxiliary + generator_reward_ext * self.auxiliary
            self.rewards[agent] = reward

        if len(self.platforms) >= 4 and self.platforms[-1][1] > 500:
            self.dones = {agent: True for agent in self.agents}
        
        obs = {agent: self._observe(agent) for agent in self.agents}

        return obs, self.rewards, self.dones, {}

    def render(self, mode='human'):
        print(f"Solver position: {self.platforms[0]}")
        print(f"Goal position: ({self.goal_x}, {self.goal_y})")
        print(f"Generator last position: ({self.generator_last_x}, {self.generator_last_y})")
        print(f"Last two platforms: {self.platforms[-4:]}")


import gym
import torch
from stable_baselines3 import PPO
from gym.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import logger

register(
    id='JumpingGame-v0',
    entry_point='custom_env:JumpingGame',
)
# Create the environment
env = gym.make('JumpingGame-v0')
 



