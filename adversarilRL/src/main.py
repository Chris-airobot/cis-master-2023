# from gymnasium.wrappers import FlattenObservation
import numpy as np
from agents import Agent
from utils import *
from double_environment import DoubleEnvironment
from argparse import ArgumentParser
import wandb


def single_training(config):
    pass



def double_training(config):
    env = DoubleEnvironment()


    n_actions = env.action_space().n
    input_dims = env.observation_space().shape

    solver = Agent(n_actions, 
                   config=config,
                   input_dims=input_dims)
    
    generator = Agent(n_actions, 
                      config=config,
                      input_dims=input_dims)
    
    
    

    figure_file = 'plots/' + config['environment_type'] +'.png'
    best_score = -1000
    score_history = []
    score_generator_history = []
    score_solver_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0


    for i in range(config['episodes']):
        curr_state = env.reset()
        done = {'prisoner': False, 'helper': False}
        truncated = {'prisoner': False, 'helper': False}
        score_generator = 0
        score_solver = 0
        score = 0



        while True not in done.values() and True not in truncated.values():

            curr_state = curr_state['prisoner']
            action_1, prob_1, val_1 = generator.choose_action(curr_state)
            action_2, prob_2, val_2 = solver.choose_action(curr_state)
            actions = {'prisoner': action_1, 'helper': action_2}
            # print(f'Prisoner action: {prisoner_action_map[actions["prisoner"]] }')
            # print(f'Helper action: {helper_action_map[actions["helper"]] }')
            next_state, reward, done, truncated, _ = env.step(actions)

            env.render()
            score_generator += reward['helper']
            score_solver += reward['prisoner']
            score =  score_generator + score_solver
            generator.remember(curr_state, action_1, prob_1, val_1, reward['helper'], done['helper'], truncated['helper'])
            solver.remember(curr_state, action_2, prob_2, val_2, reward['prisoner'], done['prisoner'], truncated['prisoner'])
            
            if n_steps % config['saving_steps'] == 0:
                generator.learn()
                solver.learn()
                learn_iters += 1
            curr_state = next_state
        score_history.append(score)
        score_generator_history.append(score_generator)
        score_solver_history.append(score_solver)

        avg_score = np.mean(score_history[-100:])
        avg_generator_score = np.mean(score_generator_history[-100:])
        avg_solver_score = np.mean(score_solver_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            generator.save_models()
            solver.save_models()

        print(f'episode: {i},  score: {score}, generator_score: {avg_generator_score}, solver_score: {avg_solver_score}, time_steps: {n_steps}, learning_steps: {learn_iters}')
        

        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history,figure_file)




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
        "episodes": 300
    }

    # wandb.init(config=config_defaults, project="adversarialrl", mode="disabled")
    # config = wandb.config  # important, in case the sweep gives different values



    if args.environment_type == 'single':
        single_training(config)
    else:
        double_training(config)









