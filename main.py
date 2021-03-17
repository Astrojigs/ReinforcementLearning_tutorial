import gym
import numpy as np

env = gym.make('CartPole-v1')
obs = env.reset()

# print(obs) #[-0.04340855 -0.04825326  0.00893705  0.02182048]
#obs = [position (0=center), velocity, angle of the pole (0 = vertical), angular velocity]
'''Render the scenario'''
env.render()

# The following are other actions possible
# print(env.action_space) # Discrete(2)
'''Discrete(2) implies that the possible actions are integers 0 or 1, which represent accelerating left(0)
or right(1).'''

#Let's accelerate the cart towards the right
action = 1
obs, reward, done, info = env.step(action=action)
#(array([-0.00291647,  0.42505656, -0.04575379, -0.60914895]), 1.0, False, {})
# The step() method executes the given actions and returns four values

# Hardcoding a simple policy that accelerates left when the pole is leaning towards the left
# and accelerates right when the pole is leaning towards the right.
# We will run this policy to see the average rewards it gets over 500 episodes
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

# Let us look at the result:
print(np.mean(totals),np.std(totals),np.min(totals),np.max(totals))
# Answer: 41.818 8.908247639126339 24.0 68.0
'''Even with 500 tries, this policy never managed to keepthe pole upright for more than 68 consecutive
steps. Let's see if a neural network can come up with a new and better policy'''


# call close() method to free resources
env.close()
