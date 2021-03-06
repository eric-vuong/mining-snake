import random
from typing import List, Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Setup and deep-q network implementation based on https://keras.io/examples/rl/deep_q_network_breakout/
debug = 0 # Set to 1 to show additional information

no_training = 1 # set to 1 to run without updating weights

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Configuration paramaters for the whole setup
seed = 1337
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter

# Lowered epsilon_min to 0.01 from 0.1 because it results in self collisions
epsilon_min = 0.01  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer

# Create neural network
def create_q_model():

    # Network defined by the Deepmind paper
    # Input is 11x11 with 3 channels (rgb)
    inputs = layers.Input(shape=(11, 11, 3,))

    # Convolutions on the frames on the screen
    # Same padding on layers 2 and 3 are required to function
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu", padding='same')(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu", padding='same')(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(4, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)
    
# Create models
model = create_q_model()
model_target = create_q_model()

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
running_reward = 0 # sum of scores
episode_count = 0 # number of games played
frame_count = 0

# Number of frames to take random action and observe output
epsilon_random_frames = 10000

# Number of frames for exploration
# Lowered from 1 000 000
epsilon_greedy_frames = 20000

# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000

# How often to update the target network
# Lowered from 10000 so it updates more frequently
update_target_network = 500

# Using huber loss for stability
loss_function = keras.losses.Huber()

# End of Atari breakout code
# Counter to track number of snakes
snake_count = -1

# Function converts the current board state into an image
# and chooses a valid move using the model
def choose_move(data: dict) -> str:
        
    # Random action for exploration
    possible_moves = [0,1,2,3] # These correspond to "up", "down", "left", "right"
    global frame_count
    global epsilon_random_frames
    global epsilon
    global snake_count
    global episode_count
    frame_count += 1
    
    # Turn 0 setup
    if data['turn'] == 0:
    
        # Set number of snakes for previous turn
        snake_count = -1
        
        # add 1 episode for each first turn seen
        episode_count += 1

    # Process board data into 11x11 image with 3 channels
    # First channel is food tiles
    # Second channel is our snakes
    # Third channel is enemy snakes
    # The brightness is based on snake health and body part
    board_image = np.zeros([11,11,3])

    # Add food
    for food in data['board']['food']:
        board_image[food['y']][food['x']][0] = 1
    
    # Add our snake's parts
    i = len(data['you']['body'])
    for part in data['you']['body']:
        if i == len(data['you']['body']):
        
            # Head
            board_image[part['y']][part['x']][1] = (data['you']['health'] + 155) / 255
            i -= 1
        elif i != 1:
        
            # Body
            board_image[part['y']][part['x']][1] = (data['you']['health'] + 75) / 255
            i -= 1
        else:
        
            # Tail
            board_image[part['y']][part['x']][1] = data['you']['health'] / 255

    # Add other snake parts
    current_snakes = 0
    for snake in data['board']['snakes']:
    
        # Skip our snake
        if data['you']['name'] == snake['name']:
            continue
        current_snakes += 1
        j = len(snake['body'])
        for part in snake['body']:
            if j == len(snake['body']):
            
                # Head
                board_image[part['y']][part['x']][2] = (snake['health'] + 155) / 255
                j -= 1
            elif j != 1:
            
                # Body
                board_image[part['y']][part['x']][2] = (snake['health'] + 75) / 255
                j -= 1
            else:
            
                # Tail
                board_image[part['y']][part['x']][2] = snake['health'] / 255
    
    # Get image for debugging
    if debug == 1:
        image = np.zeros([11,11])
        for i in range(11):
            for j in range(11):
                image[i][j] = np.sum(board_image[i][j])
        print(np.flip(image, 0))
        
    # Section based on train section from https://keras.io/examples/rl/deep_q_network_breakout/
    # Choose move using epsilon-greedy
    if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
    
        # Take random action
        move = np.random.choice(possible_moves)
    else:
    
        # Predict action Q-values
        # From environment state
        state_tensor = tf.convert_to_tensor(board_image)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
        move = possible_moves[action]
    if debug == 1:
        print(f"{data['game']['id']} MOVE {data['turn']}: {move} picked from all valid options in {possible_moves}")
    
    # Decay probability of taking random action
    # After ~100 000 frames the random action drops to 0.01
    global epsilon_interval
    global epsilon_greedy_frames
    epsilon -= epsilon_interval / epsilon_greedy_frames
    epsilon = max(epsilon, epsilon_min)

    # Save actions and states in replay buffer
    action_history.append(move)
    state_history.append(board_image)
    
    # End of Atari breakout train code
    # Update next state after the first turn
    if data['turn'] != 0:
        state_next_history.append(board_image)
    done_history.append(0) # This is set to one later if it was the last turn
    
    # Update reward of previous turn
    if data['turn'] != 0:
        reward = 0
        
        # Reward for eating food
        if data['you']['health'] == 100:
            reward += 0.5
            
        # Reward when another snake dies
        if current_snakes < snake_count:
            reward += 1
            
        # Only update if there is a reward
        if reward > 0 & len(rewards_history) > 0:
        
            # Increment most recent reward
            rewards_history[-1] = reward
            
            # Track global results
            global running_reward
            running_reward += reward
            
    # Add reward for current action, which is not yet known
    rewards_history.append(0)
    
    # Update snake_count
    snake_count = current_snakes
    return move

# Function runs at the end of a game to update the model
# Returns the frame count, loss, and running reward
# Implementation based on train section in https://keras.io/examples/rl/deep_q_network_breakout/
def postgame(win):

    # Skip updating if training is disabled
    if no_training == 1:
        return 0,0,0
        
    # Add empty next board state
    state_next_history.append(np.zeros([11,11,3]))
    
    # Update reward based on win or loss
    global running_reward
    if len(rewards_history) > 0:
        rewards_history[-1] += win
        running_reward += win
        
    # Update done status
    if len(done_history) > 0:
        done_history[len(done_history) - 1] = 1
        
    report_loss = None  
    
    # Limit the state and reward history length
    if len(rewards_history) > max_memory_length:
        del rewards_history[:]
        del state_history[:]
        del state_next_history[:]
        del action_history[:]
        del done_history[:]
        
    # Update based on batch_size after each game
    if len(done_history) > batch_size:

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
        masks = tf.one_hot(action_sample, 4)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            
            # Calculate loss between new Q-value and old Q-value
            loss = loss_function(updated_q_values, q_action)
            report_loss = loss.numpy()

        # Backpropagation
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # Update every 100 games
        if episode_count % update_target_network == 0:
            # Update the the target network with new weights
            model_target.set_weights(model.get_weights())
            return frame_count, report_loss, running_reward
        
    # Return only running reward if no update was done
    return 0, 0, running_reward

# Save model
def save(path):
    print("saving to "+path)
    model.save(path)

# Open model
def open(path):
    print("Opening model")
    global model
    model = keras.models.load_model(path)
