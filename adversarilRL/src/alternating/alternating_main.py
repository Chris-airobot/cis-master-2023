import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from src.agents import Agent
from src.utils import *
from alternating_env import AlternatingEnv
from src.solver_only.single_environment import SingleEnvironment
from argparse import ArgumentParser
import wandb

os.system('clear')
# cwd = os.getcwd()
# print(f"Current working directory: {cwd}")
first = False
files = ["src/saved_files/helper.txt", "src/saved_files/solver.txt"]


if first:
    for f in files:
        file = open(f, "w")
        #convert variable to string
        str = repr(-1000)
        file.write(str)
        file.close()


def training(helper_config, solver_config):
    
    if helper_config['environment_type'] == 'single':
        env = SingleEnvironment()
    else:
        env = AlternatingEnv()
        env.current_agent = 1
        helper = Agent(13, 
                   config=helper_config,
                   input_dims=env.observation_space().shape,
                   chkpt_dir='checkpoint/' + helper_config['environment_type'],
                   name='helper')
    
        if not first:
            helper.load_models()

    env.current_agent = 0

    solver = Agent(9, 
                    config=solver_config,
                    input_dims=env.observation_space().shape,
                #    chkpt_dir='checkpoint/' + config['environment_type'],
                    chkpt_dir='checkpoint/' + helper_config['environment_type'],
                    name='solver')
    if not first:
        solver.load_models()

 
    figure_file = {
        'solver': 'plots/' + helper_config['environment_type'] +'/solver.png',
        'helper': 'plots/' + helper_config['environment_type'] + '/helper.png'
    }



    # Read best score from previous training
    f = open('src/saved_files/solver.txt', 'r')
    if f.mode=='r':
        contents= f.read()
    solver_best_score = float(contents)
    
    f = open('src/saved_files/helper.txt', 'r')
    if f.mode=='r':
        contents= f.read()
    helper_best_score = float(contents)

    score_helper_history = []
    score_solver_history = []

    memories = {
        'solver': None,
        'helper': None
    }
    n_steps = 0
    learn_iters = 0
    completed = 0
    properties = {
        'solver': {
            'score': 0,
            'saved': 0,
        },
        'helper':{
            'score': 0,
            'saved': 0,
        }
    }


    for i in range(helper_config['episodes']):
        # auxiliary = 1
        done = False
        truncated = False
        solver_action_done = False

        properties['solver']['score'] = 0
        properties['helper']['score'] = 0

        # helper first
        curr_state = env.reset()
        
        while not done and not truncated:
            env.current_agent = 1 - env.current_agent

            if env.current_agent == 0:
                action, prob, val = solver.choose_action(curr_state) 
                next_state, reward, done, truncated, info, auxiliary = env.step(action)
                memories['solver'] = [curr_state, action, prob, val, reward, done, truncated]
                solver_action_done = True

                

            else:
                action, prob, val = helper.choose_action(curr_state) 
                next_state, reward, done, truncated, info, auxiliary = env.step(action)
                memories['helper'] = [curr_state, action, prob, val, reward, done, truncated]



            # both solver and helper has executed action once
            if solver_action_done:
                
                

                # give external rewards for the helper if the solver terminate the environment
                if memories['solver'][DONE_INDEX]:
                    if "Completed" in info['solver']:
                        completed += 1
                        memories['helper'][REWARD_INDEX] += 4 * auxiliary * 0.5
                    else:
                        memories['helper'][REWARD_INDEX] += -2 * auxiliary
                else:
                    memories['helper'][REWARD_INDEX] += memories['solver'][REWARD_INDEX] * 2

                solver.remember(*memories['solver'])
                helper.remember(*memories['helper'])

                properties['solver']['score'] += memories['solver'][REWARD_INDEX]
                properties['helper']['score'] += memories['helper'][REWARD_INDEX]
                n_steps += 1

                if n_steps % helper_config['saving_steps'] == 0 and not memories['solver'][TRUNCATED_INDEX]:
                    helper.learn()
                    learn_iters += 1
                if n_steps % solver_config['saving_steps'] == 0 and not memories['solver'][TRUNCATED_INDEX]:
                    solver.learn()
                    learn_iters += 1

                solver_action_done = False



            curr_state = next_state
  
        score_helper_history.append(properties['helper']['score'])
        score_solver_history.append(properties['solver']['score'])

        avg_helper_score = np.mean(score_helper_history[-100:])
        avg_solver_score = np.mean(score_solver_history[-100:])


        if not first:
            prepare = 50
        else:
            prepare = 10

        if avg_solver_score > solver_best_score:
            if i > prepare:
                solver_best_score = avg_solver_score

                # Save the best score for next ietration of training 
                file = open("src/saved_files/solver.txt", "w")
                #convert variable to string
                str = repr(solver_best_score)
                file.write(str)
                properties['solver']['saved'] += 1     

                # solver.save_models()

        if avg_helper_score > helper_best_score:
            if i > prepare:
                helper_best_score = avg_helper_score

                # Save the best score for next ietration of training 
                file = open("src/saved_files/helper.txt", "w")
                #convert variable to string
                str = repr(helper_best_score)
                file.write(str)
                properties['helper']['saved'] += 1
                # helper.save_models() 

        # if i % 100 == 0:
        print(f'====================================== Episode: {i} ======================================')
        print(f'Solver_score: {properties["solver"]["score"]}, avg_score: {avg_solver_score}')
        if helper_config['environment_type'] != 'single':
            print(f'Helper_score: {properties["helper"]["score"]}, avg_score: {avg_helper_score}')
        print(f'time_steps: {n_steps}, learning_steps: {learn_iters}, completed_times: {completed}, auxiliary: {auxiliary}')
        print(f'solver_model_saving: {properties["solver"]["saved"]}, helper_model_saving: {properties["helper"]["saved"]}')
        print(f'========================================================================================\n')
    
        wandb.log({"Solver_score": avg_solver_score}, step=n_steps)
        wandb.log({"Helper_score": avg_helper_score}, step=n_steps)
        wandb.log({"Completed_times": completed}, step=n_steps)



        # # Plots
        # y = [i+1 for i in range(len(score_solver_history))]
        # plot_learning_curve(y, score_solver_history,figure_file['solver'], 'solver')

        # z = [i+1 for i in range(len(score_helper_history))]
        # plot_learning_curve(z, score_helper_history,figure_file['helper'], 'helper')





if __name__ == '__main__':



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


    helper_config ={
        "environment_type" : args.environment_type,
        "saving_steps" : 20,
        "batch_size": 32,
        "epochs": 10,
        "alpha": 2e-5, # learning rate
        "clip_ratio": 0.2,
        "gamma": 0.95,   # discount factor
        "td_lambda": 0.95,
        "episodes": 30000
    }

    solver_config ={
        "environment_type" : args.environment_type,
        "saving_steps" : 20,
        "batch_size": 32,
        "epochs": 10,
        "alpha":  2e-5, # learning rate
        "clip_ratio": 0.1,
        "gamma": 0.93,   # discount factor
        "td_lambda": 0.99,
        "episodes": 30000
    }

    run = wandb.init(
        project="CIS_Master", 
        name="alternating", 
        resume=True,
        config=solver_config,
    )

    start = time.time()
    training(helper_config, solver_config)
    end = time.time()
    print(f"Total time taken: {end-start}")
    # import pettingzoo 
    # import pettingzoo.utils as pz_utils
    # from stable_baselines3 import PPO

    # env = DoubleEnvironment()
    # # Wrap the environment using the pettingzoo utility function
    # # Train the environment using PPO
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=int(1e5))









