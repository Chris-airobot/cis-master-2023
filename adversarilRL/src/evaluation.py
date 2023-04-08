from utils import *
import numpy as np
from agents import Agent
from single_environment import SingleEnvironment
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
        default="single",
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
    figure_file = {'prisoner': 'plots/' + config['environment_type'] +'/prisoner.png',}
    chkpt_dir = 'checkpoint/' + config['environment_type']
    name='prisoner'
    model_files = [chkpt_dir+'/critic_'+name, chkpt_dir+'/actor_'+name]
    
    # Creating environment 
    env = SingleEnvironment()
    
    prisoner = Agent(env.action_space().n,
                     config=config,
                     input_dims=env.observation_space().shape,
                     chkpt_dir=chkpt_dir,
                     name=name)
    
    prisoner.load_models()
    


    verbose = True
    best_score = -1000
    score_history = []
    score_helper_history = []
    score_prisoner_history = []
    completed = 0
    avg_score = 0
    n_steps = 0

    for i in range(config['episodes']):
        done = {'prisoner': False}
        truncated = {'prisoner': False}
        curr_state = env.reset()
        scores = {'prisoner' : 0}

        while True not in done.values() and True not in truncated.values():
            
            curr_state = curr_state['prisoner']
            # if verbose:
            #     print("Before moving:")
            #     print(f'Current state is: {curr_state}')
            #     print(f'map is:')
            #     env.render()
            actions = {}
            probs = {}
            vals = {}
            action, prob, val = prisoner.choose_action(curr_state)  
            actions = {'prisoner': action}
            
            next_state, reward, done, truncated, info = env.step(actions)
            if verbose:
                print(f'Prisoner action: {prisoner_action_map[actions["prisoner"]] }')
                print(f'Reward value after taking the action: {reward}')
                print("After moving:")
                env.render()
            if info['prisoner']:
                completed +=1
            n_steps += 1
            
            scores["prisoner"] += reward['prisoner']
            # prisoner.remember(curr_state, action, prob, val, reward, done, truncated)

                    
            curr_state = next_state



        score_prisoner_history.append(scores['prisoner'])
        avg_prisoner_score = np.mean(score_prisoner_history[-100:])


        print(f'episode: {i}, prisoner_score: {avg_prisoner_score}, time_steps: {n_steps}, completed_times: {completed}')

        # y = [i+1 for i in range(len(score_prisoner_history))]
        # plot_learning_curve(y, score_prisoner_history,figure_file['prisoner'], 'prisoner')

    # for file in model_files:
    #     prefix = './checkpoint_history'
    #     x = int(completed /config["episodes"]*100)
    #     target = prefix+file[10:]+f'_{x}'
    #     shutil.copyfile(file, target)

    # 19 is the inital model that can run
    # 24 is the model trained continue on 19
    # 30 is the model trained continue on 24

    # 27 is the most recent trained one before adding the heursitic function