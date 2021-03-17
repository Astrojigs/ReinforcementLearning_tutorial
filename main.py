import gym

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



# call close() method to free resources
env.close()
