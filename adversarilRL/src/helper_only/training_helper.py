# from gymnasium.wrappers import FlattenObservation
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from src.agents import Agent
from src.utils import *
from src.helper_only.helper_env import GeneratorEnv
from argparse import ArgumentParser

os.system('clear')
# cwd = os.getcwd()
# print(f"Current working directory: {cwd}")
first = False

if first:
    file = open("src/saved_files/helper.txt", "w")
    #convert variable to string
    str = repr(-1000)
    file.write(str)
    file.close()


def training(config):
    env = GeneratorEnv()
    helper = Agent(env.action_space().n, 
                   config=config,
                   input_dims=env.observation_space().shape,
                   chkpt_dir='checkpoint/' + config['environment_type'],
                   name='helper')

    
    if not first:
        helper.load_models()

 
    figure_file = {
        'helper': 'plots/' + config['environment_type'] + '/helper.png'
    }

    
    f = open('src/saved_files/helper.txt', 'r')
    if f.mode=='r':
        contents= f.read()

    helper_best_score = float(contents)

    score_helper_history = []

    memories = []
    n_steps = 0
    learn_iters = 0
    properties = {'score': 0, 'saved': 0}


    for i in range(config['episodes']):
        done = False
        truncated = False

        properties['score'] = 0

        # helper first
        curr_state = env.reset()
        
        while not done and not truncated:
            # env.current_agent = 1 - env.current_agent


            action, prob, val = helper.choose_action(curr_state) 
            next_state, reward, done, truncated, _ = env.step(action)
            memories = [curr_state, action, prob, val, reward, done, truncated]

            helper.remember(*memories)

            properties['score'] += memories[REWARD_INDEX]
            n_steps += 1

            if n_steps % config['saving_steps'] == 0 and not memories[TRUNCATED_INDEX]:
                helper.learn()
                learn_iters += 1




            curr_state = next_state
  
        score_helper_history.append(properties['score'])

        avg_helper_score = np.mean(score_helper_history[-100:])


        if not first:
            prepare = 50
        else:
            prepare = 10

        if avg_helper_score > helper_best_score:
            if i > prepare:
                helper_best_score = avg_helper_score

                # Save the best score for next ietration of training 
                file = open("src/saved_files/helper.txt", "w")
                #convert variable to string
                str = repr(helper_best_score)
                file.write(str)
                properties['saved'] += 1
                helper.save_models() 

        
        print(f'====================================== Episode: {i} ======================================')
        print(f'Helper_score: {properties["score"]}, avg_score: {avg_helper_score}')
        print(f'time_steps: {n_steps}, learning_steps: {learn_iters}')
        print(f'solver_model_saving: {properties["saved"]}, helper_model_saving: {properties["saved"]}')
        print(f'========================================================================================\n')
        

        # Plots

        z = [i+1 for i in range(len(score_helper_history))]
        plot_learning_curve(z, score_helper_history,figure_file['helper'], 'helper')





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--environment_type",
        dest="environment_type",
        default="helper_only",
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









