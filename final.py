# This file will make use of the saved model
import tensorflow as tf
import gym
import numpy as np

env = gym.make('CartPole-v1')
obs = env.reset()

model = tf.keras.models.load_model('first_model_tf_agent_cartpole.h5')

action = tf.random.uniform([1,1])>model.predict(obs[np.newaxis])
print(action)
for i in range(1000):
    action = tf.random.uniform([1,1])>model.predict(obs[np.newaxis])
    obs, reward, done, info = env.step(int(action[0,0]))
    env.render()
    if done:
        break

env.close()
