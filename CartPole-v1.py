import gymnasium as gym
import numpy as np
import time
import keyboard
import matplotlib.pyplot as plt

# Hyperparameters
alpha = 0.05 # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Start with a higher exploration rate
epsilon_decay = 0.999  # Decay rate for epsilon
epsilon_min = 0.01  # Minimum value for epsilon
num_episodes = 10000  # Number of episodes to train
skip_episodes = 500  # Number of episodes to skip when 's' is pressed

# Create the environment
env = gym.make('CartPole-v1', render_mode='human')

# Define state space bins for discretization
state_bins = [
    np.linspace(-4.8, 4.8, 24),
    np.linspace(-4, 4, 24),
    np.linspace(-0.418, 0.418, 24),
    np.linspace(-4, 4, 24)
]

def discretize_state(state):
    """Discretize the continuous state space into discrete bins."""
    state_index = []
    for i, value in enumerate(state):
        state_index.append(np.digitize(value, state_bins[i]) - 1)
    return tuple(state_index)

# Initialize Q-table with small random values
q_table = np.random.uniform(low=-0.01, high=0.01, size=(tuple(len(bins) + 1 for bins in state_bins) + (env.action_space.n,)))

# Debounce mechanism
s_pressed = False

def check_skip():
    global s_pressed  # Use the global keyword to modify the global variable
    if keyboard.is_pressed('s') and not s_pressed:  # Check if 's' is pressed and debounce flag is not set
        s_pressed = True  # Set the debounce flag
        return True
    elif not keyboard.is_pressed('s'):
        s_pressed = False  # Reset the debounce flag when 's' is released
    return False

def run_episode(render_mode):
    state = discretize_state(env.reset()[0])
    done = False
    total_reward = 0

    while not done:
        if render_mode:
            env.render()

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, truncated, info = env.step(action)
        next_state = discretize_state(next_state)

        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += alpha * td_error

        total_reward += reward
        state = next_state
    
    return total_reward

# Q-learning algorithm with skip functionality and performance tracking
episode = 0
rewards = []
while episode < num_episodes:
    if check_skip():  # Use the check_skip function to handle skip logic
        print(f"Skipping {skip_episodes} episodes...")
        env.close()  # Close the current rendering window
        env = gym.make('CartPole-v1')  # Recreate the environment without rendering

        for _ in range(skip_episodes):
            if episode < num_episodes:
                episode += 1
                epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Adjust epsilon as if the episodes were run
                reward = run_episode(render_mode=False)  # Run episode without rendering
                rewards.append(reward)  # Append the reward to the list
            else:
                break
        
        env.close()  # Close the non-rendering environment
        env = gym.make('CartPole-v1', render_mode='human')  # Recreate the environment with rendering
    else:
        total_reward = run_episode(render_mode=True)
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total reward: {total_reward}")
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode += 1

# Plot the rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()

# Close the environment
env.close()
