# Week 1:
- Building env, up to "step" function.
- Notes for week 2 meeting:
    1. They are using the Unity to build the environment, which I am not very familiary with. I tried some built-in environments, but it will take some time to learn to build custom environment since it's using c#. Instead, I tried to build through __pettingzoo__ in python, haven't finished, but should be done soon. But need some time to debug or whatever
    2. The reason I didn't use existing games from OpenAI gym is that their games are a bit different. Most of games are either competitive or collaborative, their games are semi-collaborative, which means a bit unique.
    3. They are using PPO algorithm to train, which I believe should be a __built-in algorithm in Unity ML-agent package__. But as I said, I tried to build it through python, so I implemented the PPO myself. There should be some __existing packages, but they are more like default version. They use the self-play method__, so i thought it would be better for me to implement the algorithm myself to have a better understanding. I already implemented the default version of PPO, haven't tried the self-play yet, thought I would do that after the environment has been built.
    4. Their paper is not very clear about their algorithms to me. Maybe it's because I don't really know PPO and unity, these sort of things. __They didn't say what PPO version they are using, since they are using the algorithm directly from Unity ML-agents__ like normally there are two versions, one is PPO penalty and the other is PPO clip. They also didn't really specify all the hyperparameters, which I may tune the algorithms myself later.
    5. There seems like two ways, __one is learing unity__, build the environment there and using the PPO the paper is using. __The other is building the environment myself in python__, and use the algorithm either from my end or existing libraries but probably the algorithm may need take some time.

# Week 2:
### __*Adversarial Reinforcement Learning for Proceduarl Content Generation*__
__This [paper]((https://arxiv.org/pdf/2103.04847.pdf)) is aim to solve the poor generalization problem for traditional RL agents.__
## Paper Notes:
- Approach they proposed: ARLPCG---procedurally generates and tests previously unseen environments with an auxiliary input as a control variable.
- Two agents:
    - Generator: PCG RL agent, receives a reward signal __(external reward) based on the solver's performance__ which encourages the environment design to be challenging but not impossible. Another part is from __its own actions (internal reward)__, but they didn't say how they actually set this. The formal reward function for the generator is
    $$
        r = \sum^n_{i=0} r_{int}\lambda_{A_i}\alpha_i + r_{ext}\sum^n_{i=0}\lambda_{A_i}\beta_i
    $$
    where $\lambda_{A_i}$ is the auxiliary input [-1,1], $r_{int}/r_{ext}$ are the internal/external rewards, $\alpha_i, \beta_i$ are weighting factors.
    - Solver: RL agent
- Auxiliary inputs: This is for generator, it renders the level creation controllable to a certain degree. It helps the solver achieves better generalization through generator's generated challenges, it also helps the generator to generate novel and solvable environments. 
- Training: They used PPO-clip with self-play. Training numbers between swich:
$$
Solver : Generator \rightarrow 1 : 10
$$


## Generator Reward Examples
- when solver fails, it receives a reward of $\lambda_{A_i} \times 10 $
- For an additional experiment:
    - If $\lambda_{A_i} < 0$, a positive reward is added for each time step if the vehicle is above a certain threshold above ground
    - If $\lambda_{A_i} = 1$, reward the generator if solver is approaching the goal
- In both environments, the generator receives a small negative reward per step when the auxiliary input is negative. (To force the generator to create an environment that the solver either finishes or fails fast.)
- Independent of the auxiliary input, __the generator receives an incremental reward for the solver progress.__ (Gets reward when the solver approaching the goal) 


## Training Results:
1. Fixed: Like Atari game suite
2. PCG: Generated based on a set of rules and random variables. 



# Week 4:
## Interactive way of training:
- Links: [paper](https://arxiv.org/pdf/2110.03316.pdf), [code](https://github.com/robot-learning-freiburg/CEILing/tree/main)

__Main idea is utilising the humans' feedback during training to improve performance.__ The proposed method: Corrective and evaluative Interactive Learning (CEILing).  

They used two types of feedback:
- Evaluative feedback $q$ : a scalar value $\in [0,1]$ that the human provides to the agent indicating the quality of the current agent behavior. 
    - Initial value is 1, and the good action without correction is 1
    - If the agent performs unsatisfactory at time $t$, and human is not able to easily correct it, $q = 0$, all subsequent states will be also labeled as 0 until the human changes it
    - If the agent performs unsatisfactory at time $t$, and human is able to easily correct it, $q = -1$
    - Evaluative feedback contribute to loss function, if the value is 0, it does not have any effects on updating the trajectory
- Corrective feedback: from their definition, it is an information directly provided to the agent about how to improve its actions. I checked their source code, their corrective feedback also has two types:
    - Fully control: ignore the action predicted from the RL network, the action will be fully based on the user input, (a list of keys to represent the robot action, like "a" goes left, "d" goes right and so on)  
    - Partially control: combine the action predicted by the RL network and the user's input in with a weight factor of 1:1 ratio, the new action equals to $0.5\times predict + 0.5 \times input$


# Week 5:
## Update the ideas:
 - Focus the human-interaction in the Adversarial agent, like swapping the game difficulty level
 - Come up with the module diagram of possible human-interven actions in the environment

 ## Environment Updates:
 1. Prisoner should have actions of up/down/left/right one/two block(s), which is size of 8
 2. Generator has actions of generate two block(s) up/down/left/right in a row or one block away from the standing point, which is size of 8


# Week 6:
## Evaluation Method
- Single agent only with fixed start point (0,0)  
- Goal is randomly generated in the center, both x, y are [2,5], integer only.
- The map is newly generated, and guarenteed that there will be a solution

## Solver only implementation Updates
 - Idea 1: Map only contains bridges, varying goal position
    - The training result is quite good, (method is Mahattan distance only) but the agent is not good since it does not know the case when there are traps
- Idea 2: Dynamic map, each episode has a different random map
    - The training result is very bad, potentially because that the reward function is too simple
- Idea 3: Randomly generated a map, and store it into a file. Then test the trained agent on dynamic generated maps, the value is __12.9%__. (Training 10 rounds, 300 episodes each)

## Next step is the implementation of the helper
- Potential idea could be:
    - Put the trained agent inside
    - Let the auxiliary input equal to 1, i.e. for the easy mode generation
    - Map is all zeros
    - Check if the generator will generates "easy" mode for the solver

- Next could be that let the auxiliary input equal to -1, for the hard mode

# Week 7:
- Create an environment that contains the generator only to see if the reward setting is working (the solve reward part could be use some artificial values, i.e. a trained well agent or bad agent, something like this.)
    - More specifically: give generator random choosen reward values from a list to represent the reward comes from the solver. Like a list of 10, 7 values are 0.1, 3 values are -0.1, something like this.
- Got blocks right now, consider doing the implementation of generating the whole map first.

- Setting seems right now: 
    - The sepearted helper generate *hard* maps when the auxiliary input is -1.
    - Next is trying the *simple* map, and in the end, combine everything together


# Week 8:
- Finalize the whole system environment
- Do the dynamic auxiliary input thing
- Consider about the experimental section
    - Experiments
        -  Random Map + Solver 
        -  Helper generates whole map + Solver
        -  Helper + Solver alternating Markov without human dynamic auxiliary input
        - Helper + Solver alternating Markov with human-intervention
    - Matrices
        - Convergence (training time, success rate)
        - Performance
        - $\frac{\text{\# of blocks in optimal path}}{\text{\# of blocks in the whole map}}$



### New idea:
- The evaluation environment, should be randomly genereated path to the goal
- The solver only training environment, should be one fixed path only to the goal

### Plan:
- Train the dynamic environment
    - 30000 episodes per round, 6 for now
- Consider about the evaluation metrices
- Perform experiments in above


# Week 9:
- Try with different starting positions
- Try with fixed reward for different steps
- Try the idea of swapping the difficulities based on the performance of the solver 