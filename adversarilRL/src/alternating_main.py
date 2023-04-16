# from gymnasium.wrappers import FlattenObservation
import numpy as np
from agents import Agent
from utils import *
from alternating_env import AlternatingEnv
from single_environment import SingleEnvironment
from argparse import ArgumentParser
import os

os.system('clear')
# cwd = os.getcwd()
# print(f"Current working directory: {cwd}")
first = False


def training(config):
    
    if config['environment_type'] == 'single':
        env = SingleEnvironment()
    else:
        env = AlternatingEnv()
        helper = Agent(env.action_space().n, 
                   config=config,
                   input_dims=env.observation_space().shape,
                   chkpt_dir='checkpoint/' + config['environment_type'],
                   name='helper')
    
        if not first:
            helper.load_models()

    prisoner = Agent(env.action_space().n, 
                               config=config,
                               input_dims=env.observation_space().shape,
                            #    chkpt_dir='checkpoint/' + config['environment_type'],
                               chkpt_dir='checkpoint/' + config['environment_type'],
                               name='prisoner')
    if not first:
        prisoner.load_models()
 
    figure_file = {
        'prisoner': 'plots/' + config['environment_type'] +'/prisoner.png',
        'helper': 'plots/' + config['environment_type'] + '/helper.png'
    }

    # Read best score from previous training
    f = open('src/saved_files/prisoner.txt', 'r')
    if f.mode=='r':
        contents= f.read()
    prisoner_best_score = float(contents)
    
    f = open('src/saved_files/helper.txt', 'r')
    if f.mode=='r':
        contents= f.read()
    helper_best_score = float(contents)

    score_helper_history = []
    score_prisoner_history = []

    memories = {
        'prisoner': None,
        'helper': None
    }
    n_steps = 0
    learn_iters = 0
    properties = {
        'prisoner': {
            'score': 0,
            'saved': 0,
        },
        'helper':{
            'score': 0,
            'saved': 0,
        }
    }


    for i in range(config['episodes']):
        done = False
        truncated = False
        prisoner_action_done = False

        properties['prisoner']['score'] = 0
        properties['helper']['score'] = 0

        # helper first
        curr_state = env.reset()
        
        while not done and not truncated:
            env.current_agent = 1 - env.current_agent

            if env.current_agent == 0:
                action, prob, val = prisoner.choose_action(curr_state) 
                next_state, reward, done, truncated, _ = env.step(action)
                memories['prisoner'] = [curr_state, action, prob, val, reward, done, truncated]
                prisoner_action_done = True

            else:
                action, prob, val = helper.choose_action(curr_state) 
                next_state, reward, done, truncated, _ = env.step(action)
                memories['helper'] = [curr_state, action, prob, val, reward, done, truncated]



            # both prisoner and helper has executed action once
            if prisoner_action_done:
                
                

                # give external rewards for the helper if the prisoner terminate the environment
                if memories['prisoner'][DONE_INDEX]:
                    memories['helper'][REWARD_INDEX] += -2 * -1
                else:
                    memories['helper'][REWARD_INDEX] += memories['prisoner'][REWARD_INDEX] * 2

                prisoner.remember(*memories['prisoner'])
                helper.remember(*memories['helper'])

                properties['prisoner']['score'] += memories['prisoner'][REWARD_INDEX]
                properties['helper']['score'] += memories['helper'][REWARD_INDEX]
                n_steps += 1

                if n_steps % config['saving_steps'] == 0 and not memories['prisoner'][TRUNCATED_INDEX]:
                    prisoner.learn()
                    helper.learn()
                    learn_iters += 1

                prisoner_action_done = False



            curr_state = next_state
  
        score_helper_history.append(properties['helper']['score'])
        score_prisoner_history.append(properties['prisoner']['score'])

        avg_helper_score = np.mean(score_helper_history[-100:])
        avg_prisoner_score = np.mean(score_prisoner_history[-100:])


        if not first:
            prepare = 50
        else:
            prepare = 10

        if avg_prisoner_score > prisoner_best_score:
            if i > prepare:
                prisoner_best_score = avg_prisoner_score

                # Save the best score for next ietration of training 
                file = open("src/saved_files/prisoner.txt", "w")
                #convert variable to string
                str = repr(prisoner_best_score)
                file.write(str)
                properties['prisoner']['saved'] += 1     

                prisoner.save_models()

        if avg_helper_score > helper_best_score:
            if i > prepare:
                helper_best_score = avg_helper_score

                # Save the best score for next ietration of training 
                file = open("src/saved_files/helper.txt", "w")
                #convert variable to string
                str = repr(helper_best_score)
                file.write(str)
                properties['helper']['saved'] += 1
                helper.save_models() 

        
        print(f'====================================== Episode: {i} ======================================')
        print(f'Prisoner_score: {properties["prisoner"]["score"]}, avg_score: {avg_prisoner_score}')
        if config['environment_type'] != 'single':
            print(f'Helper_score: {properties["helper"]["score"]}, avg_score: {avg_helper_score}')
        print(f'time_steps: {n_steps}, learning_steps: {learn_iters}')
        print(f'prisoner_model_saving: {properties["prisoner"]["saved"]}, helper_model_saving: {properties["helper"]["saved"]}')
        print(f'========================================================================================\n')
        

        # Plots
        y = [i+1 for i in range(len(score_prisoner_history))]
        plot_learning_curve(y, score_prisoner_history,figure_file['prisoner'], 'prisoner')

        z = [i+1 for i in range(len(score_helper_history))]
        plot_learning_curve(z, score_helper_history,figure_file['helper'], 'helper')





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
        "alpha": 5e-4, # learning rate
        "clip_ratio": 0.2,
        "gamma": 0.95,   # discount factor
        "td_lambda": 0.95,
        "episodes": 750
    }


    training(config)
    # import pettingzoo 
    # import pettingzoo.utils as pz_utils
    # from stable_baselines3 import PPO

    # env = DoubleEnvironment()
    # # Wrap the environment using the pettingzoo utility function
    # # Train the environment using PPO
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=int(1e5))









