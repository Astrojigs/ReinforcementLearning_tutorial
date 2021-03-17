# Code Description
The code in the **main.py file**  contains description on the topic of reinforcement learning given in the Oreilly Book.

The example is: Cartpole

The observations(obs) are *position*, *velocity*, *angle of pole*, *angular velocity* resp.

`env` and `obs` are from **gym**

One can list all the environments possible through OpenAI gym, using;

`gym.envs.registry.all()`

```
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1
totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards+=reward
        if done:
            break
    totals.append(episode_rewards)
```

`print(np.mean(totals),np.std(totals),np.min(totals),np.max(totals))`
`Answer: 41.818 8.908247639126339 24.0 68.0`

Even with **500** tries, this policy never managed to keep the pole upright for more than 68 consecutive steps.

 Let's see if a neural network can come up with a new and better policy

 Create a neural network policy; observations as input and it will output actions to be executed.

 It will estimate the probability of each action.

 In case of CartPole environment there are only two possible actions (left or right), so we only need
 one output neuron. I will output the probability *'p'* of action 0 (left), and of course there
 probability of actions 1 (right) will be *'1-p'*.

 For example: if it outputs **0.7**, then we will pick action 0 with 70% probability, or
 action 1 with 30% probability.

 # Making the neural network:
```
 # Code for neural network policy:
 n_inputs = 4 # == env.observation_space.shape[0]
 # Creating model
 model = tf.keras.models.Sequential()
 model.add(tf.keras.layers.Dense(5, activation='elu', input_shape=[n_inputs]))
 model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
 ```

 <ins>**Note**</ins>: If there were more than two possible actions, there would be one output neuron per action, and we would use the softmax activation function instead.

 If we knew what the best action was at each step, we could train the neural network as usual, by minimizing the cross entropy between the estimated probability distribution and the target distribution. It would be a regular supervised learning.

 However, in Reinforcement Learning the only guidance the agent gets is through rewards, and the rewards are typically sparse and delayed.
 ( for more information, see page 619)

 Tackling the problem of delay, a common strategy is to evaluate an action based on the sum of all the rewards that come after it, usually applying a **discount factor** y (gamma) at each step.

 This sum of discounted rewards is called the *action's return*.

 Consider the example; if an agent decides to right three times in a row and gets +10 reward after the first step, 0 after the second step, and finally -50 after the third step, then assuming we use a discount factor of `y=0.8`, the first actions will have a return of `10 + y x 0 + y^2 x (-50) = -22`. If the discount factor is close to 0, then future rewards won't count for much compared to immediate rewards. Conversely, if the discount factor is close to 1, then rewards far into the future will count almost as much as the immediate reward. *Typical discount factors vary from 0.9 to 0.99*. With a discount factor of 0.95, rewards 13 steps into the future count roughly for half as much as immediate rewards (since 0.95^13 ~ 0.5), while with a discount factor of 0.99, rewards 69 steps into the future count for half as much as immediate rewards. In the CartPole environment, actions have fairly short-term effects, so choosing a discount factor of 0.95 seems reasonable.
