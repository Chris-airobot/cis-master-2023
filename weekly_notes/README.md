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

