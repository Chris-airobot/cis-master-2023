import gym
import numpy as np
from agents import Agent
from utils import plot_learning_curve
from custom_environment import CustomEnvironment


if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    env = CustomEnvironment()

    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    n_actions = env.action_space().n
    # n_actions = env.action_space.n
    # print(f"n_actions is {n_actions}")
    input_dims = env.observation_space().shape
    # print(input_dims)

    # agent = Agent(n_actions, 
    #               batch_size=batch_size, 
    #               alpha=alpha, 
    #               n_epochs=n_epochs, 
    #               input_dims=input_dims)
    solver = Agent(n_actions, 
                  batch_size=batch_size, 
                  alpha=alpha, 
                  n_epochs=n_epochs, 
                  input_dims=input_dims)
    
    generator = Agent(n_actions, 
                  batch_size=batch_size, 
                  alpha=alpha, 
                  n_epochs=n_epochs, 
                  input_dims=input_dims)
    
    
    
    n_games = 300

    figure_file = 'plots/cartpole.png'
    best_score = -1000
    score_history = []
    score_generator_history = []
    score_solver_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0


    for i in range(n_games):
        curr_state = env.reset()
        done = {'prisoner': False, 'helper': False}
        truncated = {'prisoner': False, 'helper': False}
        score_generator = 0
        score_solver = 0
        score = 0

        count = 0
        while True not in done.values() and True not in truncated.values():
            count += 1
            curr_state = curr_state['prisoner']
            action_1, prob_1, val_1 = generator.choose_action(curr_state)
            action_2, prob_2, val_2 = solver.choose_action(curr_state)
            actions = {'prisoner': action_1, 'helper': action_2}
            # print(f'actions: {actions}')
            next_state, reward, done, truncated, info = env.step(actions)
            # print(f'terminated values: {done}')
            # print(f'truncated values: {truncated}')
            env.render()
            # quit()
            # if count == 2: quit()
            # print(f'next_state: {next_state}')
            # print(f"are you done? {False not in truncated}")
            # print(f'now, the count is {count}')
            
            score_generator += reward['helper']
            score_solver += reward['prisoner']
            score =  score_generator + score_solver
            # print(f'reward: {reward}')
            # print(f'done: {done}')
            # print(f'truncated: {truncated}')
            generator.remember(curr_state, action_1, prob_1, val_1, reward['helper'], done['helper'], truncated['helper'])
            solver.remember(curr_state, action_2, prob_2, val_2, reward['prisoner'], done['prisoner'], truncated['prisoner'])
            
            if n_steps % N == 0:
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






