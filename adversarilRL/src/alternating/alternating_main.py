# from gymnasium.wrappers import FlattenObservation
import numpy as np
from agents import Agent
from utils import *
from alternating_env import AlternatingEnv
from src.solver_only.single_environment import SingleEnvironment
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
        generator = Agent(env.action_space().n, 
                   config=config,
                   input_dims=env.observation_space().shape,
                   chkpt_dir='checkpoint/' + config['environment_type'],
                   name='generator')
    
        if not first:
            generator.load_models()

    solver = Agent(env.action_space().n, 
                               config=config,
                               input_dims=env.observation_space().shape,
                            #    chkpt_dir='checkpoint/' + config['environment_type'],
                               chkpt_dir='checkpoint/' + config['environment_type'],
                               name='solver')
    if not first:
        solver.load_models()
 
    figure_file = {
        'solver': 'plots/' + config['environment_type'] +'/solver.png',
        'generator': 'plots/' + config['environment_type'] + '/generator.png'
    }

    # Read best score from previous training
    f = open('src/saved_files/solver.txt', 'r')
    if f.mode=='r':
        contents= f.read()
    solver_best_score = float(contents)
    
    f = open('src/saved_files/generator.txt', 'r')
    if f.mode=='r':
        contents= f.read()
    generator_best_score = float(contents)

    score_generator_history = []
    score_solver_history = []

    memories = {
        'solver': None,
        'generator': None
    }
    n_steps = 0
    learn_iters = 0
    properties = {
        'solver': {
            'score': 0,
            'saved': 0,
        },
        'generator':{
            'score': 0,
            'saved': 0,
        }
    }


    for i in range(config['episodes']):
        done = False
        truncated = False
        solver_action_done = False

        properties['solver']['score'] = 0
        properties['generator']['score'] = 0

        # generator first
        curr_state = env.reset()
        
        while not done and not truncated:
            env.current_agent = 1 - env.current_agent

            if env.current_agent == 0:
                action, prob, val = solver.choose_action(curr_state) 
                next_state, reward, done, truncated, _ = env.step(action)
                memories['solver'] = [curr_state, action, prob, val, reward, done, truncated]
                solver_action_done = True

            else:
                action, prob, val = generator.choose_action(curr_state) 
                next_state, reward, done, truncated, _ = env.step(action)
                memories['generator'] = [curr_state, action, prob, val, reward, done, truncated]



            # both solver and generator has executed action once
            if solver_action_done:
                
                

                # give external rewards for the generator if the solver terminate the environment
                if memories['solver'][DONE_INDEX]:
                    memories['generator'][REWARD_INDEX] += -2 * -1
                else:
                    memories['generator'][REWARD_INDEX] += memories['solver'][REWARD_INDEX] * 2

                solver.remember(*memories['solver'])
                generator.remember(*memories['generator'])

                properties['solver']['score'] += memories['solver'][REWARD_INDEX]
                properties['generator']['score'] += memories['generator'][REWARD_INDEX]
                n_steps += 1

                if n_steps % config['saving_steps'] == 0 and not memories['solver'][TRUNCATED_INDEX]:
                    solver.learn()
                    generator.learn()
                    learn_iters += 1

                solver_action_done = False



            curr_state = next_state
  
        score_generator_history.append(properties['generator']['score'])
        score_solver_history.append(properties['solver']['score'])

        avg_generator_score = np.mean(score_generator_history[-100:])
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

        if avg_generator_score > generator_best_score:
            if i > prepare:
                generator_best_score = avg_generator_score

                # Save the best score for next ietration of training 
                file = open("src/saved_files/generator.txt", "w")
                #convert variable to string
                str = repr(generator_best_score)
                file.write(str)
                properties['generator']['saved'] += 1
                generator.save_models() 

        
        print(f'====================================== Episode: {i} ======================================')
        print(f'Prisoner_score: {properties["solver"]["score"]}, avg_score: {avg_solver_score}')
        if config['environment_type'] != 'single':
            print(f'Helper_score: {properties["generator"]["score"]}, avg_score: {avg_generator_score}')
        print(f'time_steps: {n_steps}, learning_steps: {learn_iters}')
        print(f'solver_model_saving: {properties["solver"]["saved"]}, generator_model_saving: {properties["generator"]["saved"]}')
        print(f'========================================================================================\n')
        

        # Plots
        y = [i+1 for i in range(len(score_solver_history))]
        plot_learning_curve(y, score_solver_history,figure_file['solver'], 'solver')

        z = [i+1 for i in range(len(score_generator_history))]
        plot_learning_curve(z, score_generator_history,figure_file['generator'], 'generator')





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









