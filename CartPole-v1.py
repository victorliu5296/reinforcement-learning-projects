import gymnasium as gym
import time 

# Create the environment with the specified render mode
env = gym.make('CartPole-v1', render_mode='human')

# Initialize variables
state = env.reset()[0]  # reset now returns a tuple (state, info)
done = False
total_reward = 0

# Main loop
while not done:
    env.render()

    # Take a random action
    action = env.action_space.sample()

    # Step the environment
    next_state, reward, done, truncated, info = env.step(action)

    total_reward += reward
    state = next_state

print(f"Total reward: {total_reward}")

# Wait for 5 seconds before closing
time.sleep(5)

env.close()
