# This file will make use of the saved model
import tensorflow as tf
model = tf.keras.models.load_model('first_model_tf_agent_cartpole.h5')
