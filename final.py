# This file will make use of the saved model
import tensorflow as tf
import gym
import numpy as np

env = gym.make('CartPole-v1')
obs = env.reset()

model = tf.keras.models.load_model('first_model_tf_agent_cartpole.h5')

action = model.predict(obs[np.newaxis])
for i in range(100):
    obs, reward, done, info = env.step(int(action[0,0]))
    env.render()

env.close()
