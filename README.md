# CIS project: Adverserial Interactive Learning
## First Stage goal:
- Literature Review
- Implement the [paper's](https://arxiv.org/pdf/2103.04847.pdf) code


# Week 1:
- Building env, up to "step" function.
- Notes for week 2 meeting:
    1. They are using the Unity to build the environment, which I am not very familiary with. I tried some built-in environments, but it will take some time to learn to build custom environment since it's using c#. Instead, I tried to build through __pettingzoo__ in python, haven't finished, but should be done soon. But need some time to debug or whatever
    2. The reason I didn't use existing games from OpenAI gym is that their games are a bit different. Most of games are either competitive or collaborative, their games are semi-collaborative, which means a bit unique.
    3. They are using PPO algorithm to train, which I believe should be a __built-in algorithm in Unity ML-agent package__. But as I said, I tried to build it through python, so I implemented the PPO myself. There should be some __existing packages, but they are more like default version. They use the self-play method__, so i thought it would be better for me to implement the algorithm myself to have a better understanding. I already implemented the default version of PPO, haven't tried the self-play yet, thought I would do that after the environment has been built.
    4. Their paper is not very clear about their algorithms to me. Maybe it's because I don't really know PPO and unity, these sort of things. __They didn't say what PPO version they are using, since they are using the algorithm directly from Unity ML-agents__ like normally there are two versions, one is PPO penalty and the other is PPO clip. They also didn't really specify all the hyperparameters, which I may tune the algorithms myself later.
    5. There seems like two ways, __one is learing unity__, build the environment there and using the PPO the paper is using. __The other is building the environment myself in python__, and use the algorithm either from my end or existing libraries but probably the algorithm may need take some time.


# Week 2:
## Meeting summary:

1. Wrong plan, should think about my own game and use python to build the environment

2. Remember to be very clear about the whole paper, this should be the first priority

## Plan:
- [Paper](https://arxiv.org/pdf/2103.04847.pdf) flowchart
- Literature reviews and plan for proposal
- Implement the env
- Prepare slides