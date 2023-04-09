from utils import *
import numpy as np
from agents import Agent
from double_environment import DoubleEnvironment
from argparse import ArgumentParser
import os
import shutil


os.system('clear')
            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--environment_type",
        dest="environment_type",
        default="adversarial",
        metavar='',
        help="options: adversarial, single, collaborative, adversarial_interaction",
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
        "episodes": 300
    }
    
    # Initial settings
    figure_file = {'helper': 'plots/' + config['environment_type'] +'/helper.png',}
    chkpt_dir = 'checkpoint/' + config['environment_type']
    name1 = 'helper'
    name2 = 'prisoner'
    model_files = [chkpt_dir+'/critic_'+name1, chkpt_dir+'/actor_'+name1, 
                   chkpt_dir+'/critic_'+name2, chkpt_dir+'/actor_'+name2]
    
    # Creating environment 
    env = DoubleEnvironment()
    
    prisoner = Agent(env.action_space().n,
                     config=config,
                     input_dims=env.observation_space().shape,
                     chkpt_dir='checkpoint/' + config['environment_type'],
                     name='prisoner')

    prisoner.load_models()
    
    helper = Agent(env.action_space().n, 
                   config=config,
                   input_dims=env.observation_space().shape,
                   chkpt_dir='checkpoint/' + config['environment_type'],
                   name='helper')

    helper.load_models()

    
    agents = {'prisoner': prisoner, 'helper': helper}

    verbose = False

    score_history = []
    score_helper_history = []
    score_prisoner_history = []
    completed = 0
    avg_score = 0
    n_steps = 0

    for i in range(config['episodes']):
        done = {'prisoner': False, 'helper':False}
        truncated = {'prisoner': False, 'helper': False}
        curr_state = env.reset()
        scores = {'prisoner' : 0, 'helper': 0}

        while True not in done.values() and True not in truncated.values():
            curr_state = curr_state['prisoner']
            actions = {}
            probs = {}
            vals = {}
            for k, agent in agents.items():
                action, prob, val = agent.choose_action(curr_state)   
                actions[k] = action
                probs[k] = prob
                vals[k] = val 
            next_state, reward, done, truncated, info = env.step(actions)
            
            if verbose:
                print(f'Helper action: {helper_action_map[actions["helper"]] }')
                print(f'Helper Reward value after taking the action: {reward["helper"]}')
                print(f'Prisoner action: {prisoner_action_map[actions["prisoner"]] }')
                print(f'Prisoner Reward value after taking the action: {reward["prisoner"]}')
                print("After actions:")
                env.render()

            if info['prisoner']:
                completed +=1

            n_steps += 1
            
            # scores['prisoner'] += reward['prisoner']
            scores['helper'] += reward['helper']

            curr_state = next_state



        score_helper_history.append(scores['helper'])
        avg_helper_score = np.mean(score_helper_history[-100:])


        print(f'episode: {i}, score: {scores["helper"]} avg_helper_score: {avg_helper_score}, time_steps: {n_steps}, completed_times: {completed}')



    # for file in model_files:
    #     prefix = './checkpoint_history'
    #     x = int(completed /config["episodes"]*100)
    #     target = prefix+file[10:]+f'_{x}'
    #     shutil.copyfile(file, target)

    # 85 is the trained agent after using the new heuristic, loaded map with np.choice goal
    # 30 is the model trained before using the heuristic, loaded map with np.choice goal
    # 93 is the agent that is trained on the map with all bridges