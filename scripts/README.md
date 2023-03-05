# Code Ideas:
- The paper is using __PPO__ and a version of self-play __(alternating Markov game)__
- It is used in the __Unity ML-Agents API__

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

