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

 <u>**Note**</u>: If there were more than two possible actions, there would be one output neuron per action, and we would use the softmax activation function instead.
