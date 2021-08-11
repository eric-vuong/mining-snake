import random
from typing import List, Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration paramaters for the whole setup
seed = 1337
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer

def create_q_model():
# create and return keras nn
    # Network defined by the Deepmind paper
    # 11 11 3 is rgb
    inputs = layers.Input(shape=(11, 11, 3,))

    # Convolutions on the frames on the screen
    # issue is the kernel sizes 8 4 3, fix is setting the next 2 layers to same padding
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu", padding='same')(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu", padding='same')(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(4, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)
model = create_q_model()
model_target = create_q_model()
# create model and model_target

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
#episode_reward_history = []
#running_reward = 0
#episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
# lowed it from 10000 to 10, such that approx every tenth game it updates
update_target_network = 10
# Using huber loss for stability
loss_function = keras.losses.Huber()

def choose_move(data: dict) -> str:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: A String, the single move to make. One of "up", "down", "left" or "right".

    Use the information in 'data' to decide your next move. The 'data' variable can be interacted
    with as a Python Dictionary, and contains all of the information about the Battlesnake board 
    for each move of the game.

    """
    # Random action for exploration
    possible_moves = ["up", "down", "left", "right"]
    global frame_count
    global epsilon_random_frames
    global epsilon
    frame_count += 1
    if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
    # Take random action
      move = np.random.choice(possible_moves)
      print(f"{data['game']['id']} MOVE {data['turn']}: {move} picked from all valid options in {possible_moves}")

      return move

    # Not random
    x = np.zeros([11,11,3])

    # Put data into rgb format
# food
    for food in data['board']['food']:
      x[food['y']][food['x']][0] = 100
# self body
    i = len(data['you']['body'])
    for part in data['you']['body']:
        if i == len(data['you']['body']):
            x[part['y']][part['x']][1] = data['you']['health'] + 2
            i -= 1
        elif i != 1:
            x[part['y']][part['x']][1] = data['you']['health'] + 1
            i -= 1
        else:
            x[part['y']][part['x']][1] = data['you']['health']

      # other bodies
    for snake in data['board']['snakes']:
        # skip own body
        if data['you']['name'] == snake['name']:
            continue
        j = len(snake['body'])
        for part in snake['body']:
            if j == len(snake['body']):
                x[part['y']][part['x']][2] = snake['health'] + 2
                j -= 1
            elif j != 1:
                x[part['y']][part['x']][2] = snake['health'] + 1
                j -= 1
            else:
                x[part['y']][part['x']][2] = snake['health']
    image = np.zeros([11,11])
    for i in range(11):
      for j in range(11):
        image[i][j] = np.sum(x[i][j])
    print(np.flip(image, 0))
    # image is for display. x is the network input
    # use x to predict a move
    # possible_moves[0]. nn produces 0,1,2,3

    # POST MOVE 
    # Decay probability of taking random action
    global epsilon_interval
    global epsilon_greedy_frames
    epsilon -= epsilon_interval / epsilon_greedy_frames
    epsilon = max(epsilon, epsilon_min)
    
    # Reward needs to be done for the last Turns action
    # It isnt immediately clear what the reward is until the turn is resolved
    #episode_reward += reward

    # Save actions and states in replay buffer
    action_history.append(move)
    state_history.append(x)
    #state_next_history.append(state_next)
    done_history.append(0) # change it to 1 in post game
    # rewards
    if data['you']['health'] == 100:
      reward = 1
    else:
      reward = 0
    rewards_history.append(reward)

    return move

# run at the end of games to update
def postgame(win):
  # update reward based on win or loss
  rewards_history[len(rewards_history - 1)] = win
  # update done
  done_history[len(done_history - 1)] = 1


  # Get indices of samples for replay buffers
  indices = np.random.choice(range(len(done_history)), size=batch_size)

  # Using list comprehension to sample from replay buffer
  state_sample = np.array([state_history[i] for i in indices])
  #state_next_sample = np.array([state_next_history[i] for i in indices])
  rewards_sample = [rewards_history[i] for i in indices]
  action_sample = [action_history[i] for i in indices]
  done_sample = tf.convert_to_tensor(
      [float(done_history[i]) for i in indices]
  )

  # Build the updated Q-values for the sampled future states
  # Use the target model for stability
  future_rewards = model_target.predict(state_sample) #state_next_sample
  # Q value = reward + discount factor * expected future reward
  updated_q_values = rewards_sample + gamma * tf.reduce_max(
      future_rewards, axis=1
  )

  # If final frame set the last value to -1
  updated_q_values = updated_q_values * (1 - done_sample) - done_sample

  # Create a mask so we only calculate loss on the updated Q-values
  masks = tf.one_hot(action_sample, 4)

  with tf.GradientTape() as tape:
      # Train the model on the states and updated Q-values
      q_values = model(state_sample)

      # Apply the masks to the Q-values to get the Q-value for action taken
      q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
      # Calculate loss between new Q-value and old Q-value
      loss = loss_function(updated_q_values, q_action)

  # Backpropagation
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  
  if frame_count % update_target_network == 0:
    # update the the target network with new weights
    model_target.set_weights(model.get_weights())
    # Log details
    #template = "running reward: {:.2f} at episode {}, frame count {}"
    #print(template.format(running_reward, episode_count, frame_count))
  # Limit the state and reward history
  if len(rewards_history) > max_memory_length:
    del rewards_history[:1]
    del state_history[:1]
    #del state_next_history[:1]
    del action_history[:1]
    del done_history[:1]

