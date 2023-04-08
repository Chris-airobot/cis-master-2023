# from gymnasium.wrappers import FlattenObservation
import numpy as np
from agents import Agent
from utils import *
from double_environment import DoubleEnvironment
from single_environment import SingleEnvironment
from argparse import ArgumentParser
import os

os.system('clear')

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
        agents['helper'].load_models()

    agents['prisoner'] = Agent(env.action_space().n, 
                               config=config,
                               input_dims=env.observation_space().shape,
                               chkpt_dir='checkpoint/' + config['environment_type'],
                               name='prisoner')
    agents['prisoner'].load_models()
 
    figure_file = {
        'total': 'plots/' + config['environment_type'] + '/total.png',
        'prisoner': 'plots/' + config['environment_type'] +'/prisoner.png',
        'helper': 'plots/' + config['environment_type'] + '/helper.png'
    }

    # Read best score from previous training
    f = open('Python.txt', 'r')
    if f.mode=='r':
        contents= f.read()
        
    best_score = float(contents)
    changed = False
    score_history = []
    score_helper_history = []
    score_prisoner_history = []
    learn_iters = 0
    # avg_score = 0
    n_steps = 0

    for i in range(config['episodes']):
        done = {}
        truncated = {}
        curr_state = env.reset()
        for key in agents.keys():
            done[key] = False
            truncated[key] = False
        scores = {'prisoner' : 0, 'helper' : 0}
        
        total_score = 0

        # print('Original Map')
        # env.render()
        
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
            # print(f'raw action value is: {actions}') 
            # print(f'Prisoner action: {prisoner_action_map[actions["prisoner"]] }')
            next_state, reward, done, truncated, infos = env.step(actions)
            # print(f'Reward value after taking the action: {reward}')
            n_steps += 1
            # env.render()
            
            for k, agent in agents.items():
                scores[k] += reward[k]
                total_score += scores[k]
                agent.remember(curr_state, actions[k], probs[k], vals[k], reward[k], done[k], truncated[k])
                        
                if n_steps % config['saving_steps'] == 0 and not truncated['prisoner']:

                    # print("you learned!")
                    # print(f'Is it time out? {infos["prisoner"]}')
                    agent.learn()
                    

                    learn_iters += 1 if config['environment_type'] == 'single' else 0.5
            curr_state = next_state
        # quit()
        # score_history.append(total_score)
        # score_helper_history.append(scores['helper'])
        score_prisoner_history.append(scores['prisoner'])

        # avg_score = np.mean(score_history[-100:])
        avg_helper_score = np.mean(score_helper_history[-100:])
        avg_prisoner_score = np.mean(score_prisoner_history[-100:])

        # if i > 100 and not changed:
        #     best_score = avg_prisoner_score
        #     changed = True

        if avg_prisoner_score > best_score:
            if i > 50:
                # print(f'your best score is: {best_score}')
                # print(f'and your current score is: {avg_prisoner_score}')
                best_score = avg_prisoner_score

                # Save the best score for next ietration of training 
                file = open("Python.txt", "w")
                #convert variable to string
                str = repr(best_score)
                file.write(str)

                changed = True
                
                for _, agent in agents.items():
                    agent.save_models()

        print(f'episode: {i}, score: {scores["prisoner"]}, avg_score: {avg_prisoner_score}, time_steps: {n_steps}, learning_steps: {learn_iters}')
        

        # Plots
        x = [i+1 for i in range(len(score_history))]
       
        plot_learning_curve(x, score_history,figure_file['total'], 'total')

        y = [i+1 for i in range(len(score_prisoner_history))]
        plot_learning_curve(y, score_prisoner_history,figure_file['prisoner'], 'prisoner')

        z = [i+1 for i in range(len(score_helper_history))]
        plot_learning_curve(z, score_helper_history,figure_file['helper'], 'helper')

    print(f'best score is: {best_score}')



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
        "alpha": 3e-4, # learning rate
        "clip_ratio": 0.2,
        "gamma": 0.99,   # discount factor
        "td_lambda": 0.95,
        "episodes": 750
    }


    training(config)









