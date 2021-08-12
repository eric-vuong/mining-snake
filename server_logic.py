import random
from typing import List, Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

debug=0

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
# Configuration paramaters for the whole setup
seed = 1337
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
# Lowered min to 0.01 from 0.1 because it results in self collisions
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
epsilon_random_frames = 500
# Number of frames for exploration
# lowered it from 1 000 000 to 100 000
epsilon_greedy_frames = 100000
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000

# How often to update the target network
# lowered it from 10000 to 100, so it updates every 100 games
update_target_network = 100
# Using huber loss for stability
loss_function = keras.losses.Huber()

# Counter to track number of snakes
snake_count = -1

# Choose a move given the current game information
def choose_move(data: dict) -> str:
        
    # Random action for exploration
    possible_moves = [0,1,2,3] #These correspond to ["up", "down", "left", "right"]
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
    # Health is represented as the brightness of the snake
    x = np.zeros([11,11,3])

    # Add food
    for food in data['board']['food']:
        x[food['y']][food['x']][0] = 100
    
    # Add own body parts
    i = len(data['you']['body'])
    for part in data['you']['body']:
        if i == len(data['you']['body']):
            # Head
            x[part['y']][part['x']][1] = data['you']['health']
            i -= 1
        elif i != 1:
            # Body
            x[part['y']][part['x']][1] = data['you']['health'] + 1
            i -= 1
        else:
            # Tail
            x[part['y']][part['x']][1] = data['you']['health'] + 2

    # Add other snakes
    current_count = 0
    for snake in data['board']['snakes']:
        # Skip own body
        if data['you']['name'] == snake['name']:
            continue
        current_count += 1
        j = len(snake['body'])
        for part in snake['body']:
            if j == len(snake['body']):
                # Head
                x[part['y']][part['x']][2] = snake['health']
                j -= 1
            elif j != 1:
                # Body
                x[part['y']][part['x']][2] = snake['health'] + 1
                j -= 1
            else:
                # Tail
                x[part['y']][part['x']][2] = snake['health'] + 2
    
    # Get image for debugging
    if debug == 1:
        image = np.zeros([11,11])
        for i in range(11):
            for j in range(11):
                image[i][j] = np.sum(x[i][j])
        print(np.flip(image, 0))

    # Choose move
    if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
        # Take random action
        move = np.random.choice(possible_moves)
    else:
        # Predict action Q-values
        # From environment state
        state_tensor = tf.convert_to_tensor(x)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
        move = possible_moves[action]

    if debug == 1 : print(f"{data['game']['id']} MOVE {data['turn']}: {move} picked from all valid options in {possible_moves}")
    
    # Decay probability of taking random action
    # After ~100 000 frames the random action drops to 0.01
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
    # Update next state after the first turn
    if data['turn'] != 0:
        state_next_history.append(x)
    done_history.append(0) # Done is set to one after the game is over
    
    # Update reward of previous action
    if data['turn'] != 0:
        reward = 0
        # Reward for eating food: 0.1
        if data['you']['health'] == 100:
            reward += 0.1
        # Reward when another snake dies: 0.5
        if current_count < snake_count:
            reward += 0.5
        # Only update if there is a reward
        if reward > 0:
            # Increment most recent reward
            rewards_history[len(rewards_history) - 1] = reward
            
            # Track global results
            global running_reward
            running_reward += reward
    # Add reward for current action 
    # Result of this action and its corresponding reward are not yet known
    rewards_history.append(0)
    
    # Update snake_count
    snake_count = current_count
    return move

# run at the end of games to update
def postgame(win):
    # Add empty next state
    state_next_history.append(np.zeros([11,11,3]))
    # Update reward based on win or loss
    if len(rewards_history) > 0:
        rewards_history[len(rewards_history) - 1] = win
        global running_reward
        running_reward += win
        
    # Update done
    if len(done_history) > 0:
        done_history[len(done_history) - 1] = 1
        
        
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

        # Backpropagation
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Update every 100 games
    if episode_count % update_target_network == 0:
        # Update the the target network with new weights
        model_target.set_weights(model.get_weights())
        return frame_count, loss.numpy(), running_reward

    # Limit the state and reward history
    if len(rewards_history) > max_memory_length:
        del rewards_history[:1]
        del state_history[:1]
        del state_next_history[:1]
        del action_history[:1]
        del done_history[:1]
    return 0, loss.numpy(), running_reward

def save(path):
    print("saving to "+path)
    model.save(path)

def open(path):
    global model
    model = keras.models.load_model(path)
