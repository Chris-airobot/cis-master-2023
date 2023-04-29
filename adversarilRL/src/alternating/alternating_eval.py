import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from src.agents import Agent
from src.utils import *
from alternating_env import AlternatingEnv
from src.solver_only.single_environment import SingleEnvironment
from argparse import ArgumentParser
import shutil

os.system('clear')
            
if __name__ == '__main__':
    verbose = False
    interactive = False
    saving = True
    test = True
    episodes = 300 if test else 1

    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--environment_type",
        dest="environment_type",
        default="alternating",
        metavar='',
        help="options: adversarial, single, collaborative, adversarial_interaction, alternating",
    )
    args = parser.parse_args()

    config ={
        "environment_type" : args.environment_type,
        "saving_steps" : 20,
        "batch_size": 5,
        "epochs": 4,
        "alpha": 0.0003, # learning rate
        "clip_ratio": 0.2,
        "gamma": 0.99,   # discount factor
        "td_lambda": 0.95,
        "episodes": episodes
    }
    
    # Initial settings
    chkpt_dir = 'checkpoint/' + config['environment_type']
    name1 = 'helper'
    name2 = 'solver'
    model_files = [chkpt_dir+'/critic_'+name1, chkpt_dir+'/actor_'+name1, 
                   chkpt_dir+'/critic_'+name2, chkpt_dir+'/actor_'+name2]
    # Creating environment 
    env = AlternatingEnv()
    env.current_agent = 1
    helper = Agent(13,
                     config=config,
                     input_dims=env.observation_space().shape,
                     chkpt_dir='checkpoint/' + config['environment_type'],
                     name='helper')

    helper.load_models()
    env.current_agent = 0
    solver = Agent(9, 
                   config=config,
                   input_dims=env.observation_space().shape,
                   chkpt_dir='checkpoint/' + config['environment_type'],
                   name='solver')

    solver.load_models()

    
    agents = {'helper': helper, 'solver': solver}


    # score_history = []
    score_solver_history = []
    score_helper_history = []
    completed = 0
    avg_score = 0
    n_steps = 0
  

    properties = {
        'helper': {
            'learn_iters': 0,
            'n_steps': 0,
            'score': 0,
            'saved': 0
        },
        'solver':{
            'learn_iters': 0,
            'n_steps': 0,
            'score': 0,
            'saved': 0
        }
    }


    for i in range(config['episodes']):
        done = False
        truncated = False

        properties['helper']['score'] = 0
        properties['solver']['score'] = 0

        # helper first
        curr_state = env.reset()
        
        while not done and not truncated:
            env.current_agent = 1 - env.current_agent

            if env.current_agent == 0:
                action, prob, val = solver.choose_action(curr_state) 
                if interactive:
                    print("What is the action of the Solver?")
                    print("0: up 1, 1: down 1, 2: left 1, 3: right 1")
                    print("4: up 2, 5: down 2, 6: left 2, 7: right 2")
                    action = int(input())

                next_state, reward, done, truncated, info = env.step(action)
                properties['solver']['score'] += reward
                properties['solver']['n_steps'] += 1

                if truncated:
                    print(f"solver reward: {reward}")
                if "Completed" in info['solver']:
                    completed += 1

                if verbose:
                    print(f'{solver_action_map[action]}, {info["solver"]}')
                    print(f'Solver Reward value after taking the action: {reward}')
                    # print("After actions:")
                    env.render()

            else:
                action, prob, val = helper.choose_action(curr_state) 
                if interactive:
                    print("What is the action of the Helper?")
                    print("0: up 2, 1: down 2, 2: left 2, 3: right 2")
                    print("4: up 1, 5: down 1, 6: left 1, 7: right 1")
                    action = int(input())

                next_state, reward, done, truncated, info = env.step(action)
                properties['helper']['score'] += reward
                properties['helper']['n_steps'] += 1

                if verbose:
                    print(f'{helper_action_map[action]}, {info["helper"]}')
                    print(f'Helper Reward value after taking the action: {reward}')
                    # print("After actions:")
                    env.render()
                
                if truncated:
                    print(f"helper reward: {reward}")


            curr_state = next_state
  
        score_solver_history.append(properties['solver']['score'])
        score_helper_history.append(properties['helper']['score'])

        avg_solver_score = np.mean(score_solver_history[-100:])
        avg_helper_score = np.mean(score_helper_history[-100:])


        print(f'episode: {i}, time_steps: {n_steps}, completed_times: {completed}')
        print(f'avg_solver_score: {avg_solver_score} avg_helper_score: {avg_helper_score}')


    if saving:
        for file in model_files:
            prefix = './checkpoint_history'
            x = int(completed /config["episodes"]*100)
            target = prefix+file[10:]+f'_{x}'
            shutil.copyfile(file, target)

    # 45 is the trained model with a potential good success rate but no time outs 
    # 97 is the model that trained good in the end, but has some time outss