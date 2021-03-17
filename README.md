# Code Description
The code in the **main.py file**  contains description on the topic of reinforcement learning given in the Oreilly Book.

The example is: Cartpole

The observations(obs) are *position*, *velocity*, *angle of pole*, *angular velocity* resp.

`env` and `obs` are from **gym**

One can list all the environments possible through OpenAI gym, using;

`gym.envs.registry.all()`

```def basic_policy(obs):
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
    totals.append(episode_rewards)```
