import gym
import numpy as np
import tensorflow as tf

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
'''Even with 500 tries, this policy never managed to keep the pole upright for more than 68 consecutive
steps. Let's see if a neural network can come up with a new and better policy'''

# Create a neural network policy; observations as input and it will output actions to be executed.
# It will estimate the probability of each action
'''
In case of CartPole environment there are only two possible actions (left or right), so we only need
one output neuron. I will output the probability 'p' of action 0 (left), and of course there
probability of actions 1 (right) will be '1-p'.

For example: if it outputs 0.7, then we will pick action 0 with 70% probability, or
action 1 with 30% probability.'''

# Code for neural network policy:
n_inputs = 4 # == env.observation_space.shape[0]
# Creating model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(5, activation='elu', input_shape=[n_inputs]))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

# Function to play a step in the environment
def play_one step(env,obs,model,loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis]) #obs[np.newaxis] and obs.reshape((1,)+ obs.shape[:]) is the same
        action = (tf.random.uniform([1,1])>left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action,tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target,left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0,0].numpy()))
    return obs, reward, done, grads

# Function to play multiple episodes:
def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads, = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads
# call close() method to free resources
env.close()
