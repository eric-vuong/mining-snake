import random
from typing import List, Dict
import numpy as np
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""

"""
This file can be a nice home for your move logic, and to write helper functions.

We have started this for you, with a function to help remove the 'neck' direction
from the list of possible moves!
"""

# TODO
#def create_q_model():
# create and return keras nn

# create model and model_target

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
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
update_target_network = 10000
# Using huber loss for stability
#loss_function = keras.losses.Huber()

def choose_move(data: dict) -> str:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: A String, the single move to make. One of "up", "down", "left" or "right".

    Use the information in 'data' to decide your next move. The 'data' variable can be interacted
    with as a Python Dictionary, and contains all of the information about the Battlesnake board 
    for each move of the game.

    """
    #my_head = data["you"]["head"]  # A dictionary of x/y coordinates like {"x": 0, "y": 0}
    #my_body = data["you"]["body"]  # A list of x/y coordinate dictionaries like [ {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0} ]
    # TODO: uncomment the lines below so you can see what this data looks like in your output!
    # print(f"~~~ Turn: {data['turn']}  Game Mode: {data['game']['ruleset']['name']} ~~~")
    x = np.zeros([3,11,11])
    

    

# food
    for food in data['board']['food']:
      x[0][food['y']][food['x']] = 1
# self body
    i = len(data['you']['body'])
    for part in data['you']['body']:
        if i == len(data['you']['body']):
            x[1][part['y']][part['x']] = data['you']['health']
            i -= 1
        elif i != 1:
            x[1][part['y']][part['x']] = data['you']['health'] + 1
            i -= 1
        else:
            x[1][part['y']][part['x']] = data['you']['health'] + 2

      # other bodies
    i = 0
    for snake in data['board']['snakes']:
        # skip own body
        if i == 0:
            i = 1
            continue
        j = len(snake['body'])
        for part in snake['body']:
            if j == len(snake['body']):
                x[2][part['y']][part['x']] = snake['health']
                j -= 1
            elif j != 1:
                x[2][part['y']][part['x']] = snake['health'] + 1
                j -= 1
            else:
                x[2][part['y']][part['x']] = snake['health'] + 2
    image = np.flip(x,1)        
    print(image)
    
    # explore random moves or predict best move with nn
    possible_moves = ["up", "down", "left", "right"]
    move = 'up'
    print(f"{data['game']['id']} MOVE {data['turn']}: {move} picked from all valid options in {possible_moves}")

    return move
# Function is run after move is returned
def postmove(move):
  # Decay probability of taking random action
  epsilon -= epsilon_interval / epsilon_greedy_frames
  epsilon = max(epsilon, epsilon_min)
  
  # Reward needs to be done for the last Turns action
  # It isnt immediately clear what the reward is until the turn is resolved
  #episode_reward += reward

  # Save actions and states in replay buffer
  action_history.append(move)
  state_history.append(state) # state can be the image
  state_next_history.append(state_next) # image of the next turn
  done_history.append(done) # ??? not sure what done is
  rewards_history.append(reward) # need to figure out reward system

# run at the end of games to update
def postgame():
  # Get indices of samples for replay buffers
  indices = np.random.choice(range(len(done_history)), size=batch_size)

  # Using list comprehension to sample from replay buffer
  state_sample = np.array([state_history[i] for i in indices])
  state_next_sample = np.array([state_next_history[i] for i in indices])
  rewards_sample = [rewards_history[i] for i in indices]
  action_sample = [action_history[i] for i in indices]
  done_sample = tf.convert_to_tensor(
      [float(done_history[i]) for i in indices]
  )

  # Build the updated Q-values for the sampled future states
  # Use the target model for stability
  future_rewards = model_target.predict(state_next_sample)
  # Q value = reward + discount factor * expected future reward
  updated_q_values = rewards_sample + gamma * tf.reduce_max(
      future_rewards, axis=1
  )

  # If final frame set the last value to -1
  updated_q_values = updated_q_values * (1 - done_sample) - done_sample

  # Create a mask so we only calculate loss on the updated Q-values
  masks = tf.one_hot(action_sample, num_actions)

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
  
  # Use game count instead of frame counts
  if frame_count % update_target_network == 0:
    # update the the target network with new weights
    model_target.set_weights(model.get_weights())
    # Log details
    template = "running reward: {:.2f} at episode {}, frame count {}"
    print(template.format(running_reward, episode_count, frame_count))

  # Limit the state and reward history
  if len(rewards_history) > max_memory_length:
    del rewards_history[:1]
    del state_history[:1]
    del state_next_history[:1]
    del action_history[:1]
    del done_history[:1]

  # Update running reward to check condition for solving
  episode_reward_history.append(episode_reward) 
  if len(episode_reward_history) > 100:
    del episode_reward_history[:1]
  running_reward = np.mean(episode_reward_history)

  episode_count += 1

