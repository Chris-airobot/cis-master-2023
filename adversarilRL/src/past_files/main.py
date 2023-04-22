# from gymnasium.wrappers import FlattenObservation
import numpy as np
from agents import Agent
from utils import *
from double_environment import DoubleEnvironment
from single_environment import SingleEnvironment
from argparse import ArgumentParser
import os

os.system('clear')

first = False


def training(config):
    agents = {}
    if config['environment_type'] == 'single':
        env = SingleEnvironment()
    else:
        env = DoubleEnvironment()
        agents['helper'] = Agent(env.action_space().n, 
                   config=config,
                   input_dims=env.observation_space().shape,
                   chkpt_dir='checkpoint/' + config['environment_type'],
                   name='helper')
    
        if not first:
            agents['helper'].load_models()

    agents['prisoner'] = Agent(env.action_space().n, 
                               config=config,
                               input_dims=env.observation_space().shape,
                            #    chkpt_dir='checkpoint/' + config['environment_type'],
                               chkpt_dir='checkpoint/' + config['environment_type'],
                               name='prisoner')
    if not first:
        agents['prisoner'].load_models()
 
    figure_file = {
        'prisoner': 'plots/' + config['environment_type'] +'/prisoner.png',
        'helper': 'plots/' + config['environment_type'] + '/helper.png'
    }

    # Read best score from previous training
    f = open('prisoner.txt', 'r')
    if f.mode=='r':
        contents= f.read()
    prisoner_best_score = float(contents)
    
    f = open('helper.txt', 'r')
    if f.mode=='r':
        contents= f.read()
    helper_best_score = float(contents)

    changed = False
    score_helper_history = []
    score_prisoner_history = []
    learn_iters = 0
    n_steps = 0
    saved = 0

    for i in range(config['episodes']):
        done = {}
        truncated = {}
        curr_state = env.reset()
        for key in agents.keys():
            done[key] = False
            truncated[key] = False
        scores = {'prisoner' : 0, 'helper' : 0}
        
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
            next_state, reward, done, truncated, _ = env.step(actions)
            n_steps += 1
            
            for k, agent in agents.items():
                scores[k] += reward[k]
                agent.remember(curr_state, actions[k], probs[k], vals[k], reward[k], done[k], truncated[k])
                        
                if n_steps % config['saving_steps'] == 0 and not truncated['prisoner'] and not truncated['helper']:

                    # print("you learned!")
                    # print(f'Is it time out? {infos["prisoner"]}')
                    agent.learn()
                    

                    learn_iters += 1 if config['environment_type'] == 'single' else 0.5
            curr_state = next_state

        score_helper_history.append(scores['helper'])
        score_prisoner_history.append(scores['prisoner'])

        avg_helper_score = np.mean(score_helper_history[-100:])
        avg_prisoner_score = np.mean(score_prisoner_history[-100:])

        # if i > 100 and not changed:
        #     best_score = avg_prisoner_score
        #     changed = True
        if not first:
            prepare = 50
        else:
            prepare = 10

        if avg_prisoner_score > prisoner_best_score:
            if i > prepare:
                prisoner_best_score = avg_prisoner_score

                # Save the best score for next ietration of training 
                file = open("prisoner.txt", "w")
                #convert variable to string
                str = repr(prisoner_best_score)
                file.write(str)

                changed = True
                saved += 1                
                agents['prisoner'].save_models()

        if avg_helper_score > helper_best_score:
            if i > prepare:
                helper_best_score = avg_helper_score

                # Save the best score for next ietration of training 
                file = open("helper.txt", "w")
                #convert variable to string
                str = repr(helper_best_score)
                file.write(str)

                changed = True
                saved += 1
                agents['helper'].save_models() 

        
        print(f'====================================== Episode: {i} ======================================')
        print(f'Prisoner_score: {scores["prisoner"]}, avg_score: {avg_prisoner_score}')
        if config['environment_type'] != 'single':
            print(f'Helper_score: {scores["helper"]}, avg_score: {avg_helper_score}')
        print(f'time_steps: {n_steps}, learning_steps: {learn_iters}, model_saving: {saved}')
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
        "alpha": 3e-4, # learning rate
        "clip_ratio": 0.2,
        "gamma": 0.99,   # discount factor
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









