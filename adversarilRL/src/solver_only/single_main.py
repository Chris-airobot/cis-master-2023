import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from src.agents import Agent
from src.utils import *
from src.solver_only.single_environment import SingleEnvironment
from argparse import ArgumentParser


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


def training(solver_config):
    
    
    env = SingleEnvironment()
    
    



    solver = Agent(9, 
                    config=solver_config,
                    input_dims=env.observation_space().shape,
                #    chkpt_dir='checkpoint/' + config['environment_type'],
                    chkpt_dir='checkpoint/' + solver_config['environment_type'],
                    name='solver')
    if not first:
        solver.load_models()


    # Read best score from previous training
    f = open('src/saved_files/solver.txt', 'r')
    if f.mode=='r':
        contents= f.read()
    solver_best_score = float(contents)
    
 

    score_solver_history = []

    memories = {
        'solver': None,
    }
    n_steps = 0
    learn_iters = 0
    completed = 0
    properties = {
        'solver': {
            'score': 0,
            'saved': 0,
        },
    }


    for i in range(solver_config['episodes']):
        # auxiliary = 1
        done = False
        truncated = False

        properties['solver']['score'] = 0

        # helper first
        curr_state = env.reset()
        
        while not done and not truncated:


            action, prob, val = solver.choose_action(curr_state) 
            next_state, reward, done, truncated, info = env.step(action)
            memories['solver'] = [curr_state, action, prob, val, reward, done, truncated]

                
                

            # give external rewards for the helper if the solver terminate the environment
            if memories['solver'][DONE_INDEX]:
                if "Completed" in info['solver']:
                    completed += 1
                    
            solver.remember(*memories['solver'])

            properties['solver']['score'] += memories['solver'][REWARD_INDEX]
            n_steps += 1
            if n_steps % solver_config['saving_steps'] == 0 and not memories['solver'][TRUNCATED_INDEX]:
                solver.learn()
                learn_iters += 1

            curr_state = next_state
  
        score_solver_history.append(properties['solver']['score'])
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

                solver.save_models()

        # if i % 100 == 0:
        print(f'====================================== Episode: {i} ======================================')
        print(f'Prisoner_score: {properties["solver"]["score"]}, avg_score: {avg_solver_score}')
        print(f'time_steps: {n_steps}, learning_steps: {learn_iters}, completed_times: {completed}')
        print(f'solver_model_saving: {properties["solver"]["saved"]}')
        print(f'========================================================================================\n')
            

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
        default="single",
        metavar='',
        help="options: adversarial, single, collaborative, adversarial_interaction, alternating",
    )
    args = parser.parse_args()



    solver_config ={
        "environment_type" : args.environment_type,
        "saving_steps" : 20,
        "batch_size": 32,
        "epochs": 10,
        "alpha":  5e-5, # learning rate
        "clip_ratio": 0.3,
        "gamma": 0.95,   # discount factor
        "td_lambda": 0.99,
        "episodes": 30000
    }


    start = time.time()
    training(solver_config)
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









