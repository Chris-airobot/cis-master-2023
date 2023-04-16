from utils import *
import numpy as np
from agents import Agent
from alternating_env import AlternatingEnv
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
        "episodes": 1
    }
    
    # Initial settings
    figure_file = {'helper': 'plots/' + config['environment_type'] +'/helper.png',}
    chkpt_dir = 'checkpoint/' + config['environment_type']
    name1 = 'helper'
    name2 = 'prisoner'
    model_files = [chkpt_dir+'/critic_'+name1, chkpt_dir+'/actor_'+name1, 
                   chkpt_dir+'/critic_'+name2, chkpt_dir+'/actor_'+name2]
    # Creating environment 
    env = AlternatingEnv()
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

    verbose = True
    interactive = False
    # score_history = []
    score_helper_history = []
    score_prisoner_history = []
    completed = 0
    avg_score = 0
    n_steps = 0

    properties = {
        'prisoner': {
            'learn_iters': 0,
            'n_steps': 0,
            'score': 0,
            'saved': 0
        },
        'helper':{
            'learn_iters': 0,
            'n_steps': 0,
            'score': 0,
            'saved': 0
        }
    }


    for i in range(config['episodes']):
        done = False
        truncated = False

        properties['prisoner']['score'] = 0
        properties['helper']['score'] = 0

        # helper first
        curr_state = env.reset()
        
        while not done and not truncated:
            env.current_agent = 1 - env.current_agent

            if env.current_agent == 0:
                action, prob, val = prisoner.choose_action(curr_state) 
                if interactive:
                    print("What is the action of the Prisoner?")
                    print("0: up 1, 1: down 1, 2: left 1, 3: right 1")
                    print("4: up 2, 5: down 2, 6: left 2, 7: right 2")
                    action = int(input())

                next_state, reward, done, truncated, info = env.step(action)
                properties['prisoner']['score'] += reward
                properties['prisoner']['n_steps'] += 1

                if verbose:
                    print(f'{prisoner_action_map[action]}, {info["prisoner"]}')
                    print(f'Prisoner Reward value after taking the action: {reward}')
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


            curr_state = next_state
  
        score_helper_history.append(properties['helper']['score'])
        # score_prisoner_history.append(properties['prisoner']['score'])

        avg_helper_score = np.mean(score_helper_history[-100:])
        # avg_prisoner_score = np.mean(score_prisoner_history[-100:])


        print(f'episode: {i}, score: {avg_helper_score} avg_helper_score: {avg_helper_score}, time_steps: {n_steps}, completed_times: {completed}')



    # for file in model_files:
    #     prefix = './checkpoint_history'
    #     x = int(completed /config["episodes"]*100)
    #     target = prefix+file[10:]+f'_{x}'
    #     shutil.copyfile(file, target)

    # 45 is the trained model with a potential good success rate but no time outs 
    # 97 is the model that trained good in the end, but has some time outss