import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import *
import numpy as np
from src.agents import Agent
from single_environment import SingleEnvironment
from argparse import ArgumentParser
import os
import shutil


os.system('clear')
            
if __name__ == '__main__':
    verbose = False
    interactive = False
    saving = False
    testing = True

    episodes = 300 if testing else 1
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--environment_type",
        dest="environment_type",
        default="alternating",
        metavar='',
        help="options: adversarial, single, collaborative, alternating",
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
    figure_file = {'solver': 'plots/' + config['environment_type'] +'/solver.png',}
    chkpt_dir = 'checkpoint/' + config['environment_type']
    name='solver'
    model_files = [chkpt_dir+'/critic_'+name, chkpt_dir+'/actor_'+name]
    
    # Creating environment 
    env = SingleEnvironment()
    
    solver = Agent(env.action_space().n,
                     config=config,
                     input_dims=env.observation_space().shape,
                     chkpt_dir=chkpt_dir,
                     name=name)
    
    solver.load_models()
    




    score_history = []
    score_solver_history = []
    memories = []
    n_steps = 0
    learn_iters = 0
    properties = {'score': 0, 'completed': 0}

    for i in range(config['episodes']):
        done = False
        truncated = False
        curr_state = env.reset()
        properties['score'] = 0
        # env.render()
        while not done and not truncated:
            actions = {}
            probs = {}
            vals = {}
            action, prob, val = solver.choose_action(curr_state)  
            if interactive:
                print("What is the action of the Helper?")
                print("0: up 2, 1: down 2, 2: left 2, 3: right 2")
                print("4: jump up 1, 5: jump down 1, 6: jump left 1, 7: jump right 1")
                print("8: up 1, 9: down 1, 10: left 1, 11: right 1")
                action = int(input())
            
            next_state, reward, done, truncated, info = env.step(action)
            if verbose:
                print(f'Helper action: {solver_action_map[action] }')
                print(f'Reward value after taking the action: {reward}')
                print("After moving:")
                env.render()

            n_steps += 1
            properties["score"] += reward
            # solver.remember(curr_state, action, prob, val, reward, done, truncated)

            if "Completed" in info['solver']:
                properties['completed'] += 1
            curr_state = next_state



        score_solver_history.append(properties["score"])
        avg_solver_score = np.mean(score_solver_history[-100:])


        print(f'episode: {i}, current_score: {properties["score"]} solver_score: {avg_solver_score}, time_steps: {n_steps}, completed: {properties["completed"]}')


    if saving:
        for file in model_files:
            prefix = './checkpoint_history'
            x = int(properties['completed'] /config["episodes"]*100)
            target = prefix+file[10:]+f'_{x}'
            shutil.copyfile(file, target)