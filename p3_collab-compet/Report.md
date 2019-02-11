# Performance Report

## Learning Algorithm
For this project, **PPO (Proximal Policy Optimization)**
was implemented given continuous nature of the problem. **PPO** is a policy gradient method.  Compared to **TRPO (Trust Region Policy Optimization)**,
PPO is more general, simpler to implement, and is known to have higher sampling efficieny.
PPO calculates gradient based on the policy likelihood ratio between old and new policy, and leverages ratio clipping
to avoid gradient explosion. [GAE (Generalized Advantage Estimation)](https://arxiv.org/abs/1506.02438) method was used to estimate the advantage function.

## Result & Plot of Rewards
The environment was solved in under 193 episodes.

```bash
Episode: 10, average score: 1.17
Episode: 20, average score: 1.39
Episode: 30, average score: 1.69
Episode: 40, average score: 2.04
Episode: 50, average score: 2.46
Episode: 60, average score: 2.95
Episode: 70, average score: 3.52
Episode: 80, average score: 4.26
Episode: 90, average score: 5.20
Episode: 100, average score: 6.32
Episode: 110, average score: 8.19
Episode: 120, average score: 10.47
Episode: 130, average score: 12.93
Episode: 140, average score: 15.65
Episode: 150, average score: 18.52
Episode: 160, average score: 21.44
Episode: 170, average score: 24.32
Episode: 180, average score: 27.04
Episode: 190, average score: 29.47
Environment solved in 193 episodes!	Average Score: 30.15
Average Score: 30.15
Elapsed time: 0:26:39.039894
```

The following figure illustrates the average score over time obtained during training:

![scores_plot.png](./scores_plot.png)

The performance of the trained agent can be viewed in the simulator using `eval_ppo.py` script:

``` bash
python eval_ppo.py
```

![reacher_e193](./../doc/gif/reacher_193e.gif)

The model was evaluated after training by loading `ppo_128x128_a64_c64_193e.pth`, and running the environment in evaluation mode (i.e. `train_mode=False`):

```bash
Episode: 1, score: 37.36899916473776
Episode: 2, score: 37.552999160625035
Episode: 3, score: 37.534499161038546
Episode: 4, score: 37.61199915930628
Episode: 5, score: 37.618799159154285
Episode: 6, score: 37.60691582608657
Episode: 7, score: 37.60857058795434
Episode: 8, score: 37.58049916001036
Episode: 9, score: 37.64861026959907
Episode: 10, score: 37.708999157138166
```

## Network Architecture
An actor-critic structure with continuous action space was used for this project. The policy consists of 3 parts, a shared hidden layers, actor, and critic.
The actor layer outputs the mean value of a normal distribution, from which the agent's action is sampled. The critic layer yields the value function.

- Shared layer:
```
Input State(33) -> Dense(128) -> LeakyReLU -> Dense(128) -> LeakyReLU*
```
- Actor and Critic layers:
```
LeakyRelu* -> Dense(64) -> LeakyRelu -> Dense(4)-> tanh -> Actor's output
LeakyReLU* -> Dense(64) -> LeakyRelu -> Dense(1) -> Critic's output
```

### Model update using PPO/GAE
The hyperparameters used during training are:

Parameter | Value | Description
------------ | ------------- | -------------
Number of Agents | 20 | Number of agents trained simultaneously
Episodes | 2000 | Maximum number of training episodes
tmax | 1000 | Maximum number of steps per episode
Epochs | 10 | Number of training epoch per batch sampling
Batch size | 128*20 | Size of batch taken from the accumulated  trajectories
Discount (gamma) | 0.99 | Discount rate 
Epsilon | 0.1 | Ratio used to clip r = new_probs/old_probs during training
Gradient clip | 10.0 | Maximum gradient norm 
Beta | 0.01 | Entropy coefficient 
Tau | 0.95 | tau coefficient in GAE
Learning rate | 2e-4 | Learning rate 
Optimizer | Adam | Optimization method

## Ideas for Future Work
0. Extensive hyperparameter optimization and analysis
1. Further analysis on PPO to study the effect of clipping and loss error variatoin including fixed and adaptive KL penalty as done in TRPO
2. Performance comparison with [A3C](https://openreview.net/pdf?id=SyZipzbCb), [DDPG](https://arxiv.org/abs/1509.02971), and [D4PG](https://arxiv.org/pdf/1602.01783.pdf)
3. Extension to other environments such as [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) 
and [Walker](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#walker)
