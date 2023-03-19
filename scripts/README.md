# Code Ideas:
- The paper is using __PPO__ and a version of self-play __(alternating Markov game)__
- It is used in the __Unity ML-Agents API__

## Reinforcement Learning Taxonomy:
### ___On-policy vs Off-policy___
Depends on whether updating Q values based on your current policy or not. Assume the current policy is a completely random policy, you are in state __s__ and make an action __a__ that leads to state __s'__. 
- Off-policy: updating $Q(s,a)$ based on the best action you can take in __s'__
- On-policy: updating $Q(s,a)$ based on the action according to your current policy

### ___Policy-based vs Value-based___
- Policy-based: build a policy and keep it in memory during learning
- Value-based: only use a value function, pick actions with the best value

### __Deterministic Policies vs Stochastic Policies__
- Deterministic: 
    - Always exists an optimal deterministic policy
    - Search space is smaller compared with stochastic
    - Practitioners prefer deterministic policies
- Stochastic:
    - Search space is continuous for stochastic policies (helps with gradient descent)
    - More robust (Choosing actions based on the distribution which could be more reasonable)
    - Naturally incorporate exploration
    - Facilitate transfer training
    - Mitigate local optima (jump out ofx the local optimal)



## Proximal Policy Optimization:
1. Actor critic methods are sensitive to perturbations
2. PPO addresses this by limiting the update to policy network
3. __Basic idea is the update on the ration of new policy to old__
4. Have to account for goodness of state (advantage)
5. Also clipping the loss function and taking lower bound with min


## Implementation ideas:
- Angle should simply represent the xy plane angle (assume z going up)
- Distance between two blocks is from one block's center to the other one's center
- Plan to use only last one block instead of two

## Environment Dependencies
- python == 3.10
- pettingzoo == 1.22.3
- gymnasium == 0.27.1

W