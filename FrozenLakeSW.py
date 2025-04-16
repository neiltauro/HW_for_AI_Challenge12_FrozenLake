import numpy as np
import gymnasium as gym  # Use gymnasium instead of gym

# Initialize environment
env = gym.make('FrozenLake-v1', is_slippery=False)  # deterministic version for clarity
state_size = env.observation_space.n
action_size = env.action_space.n

# Hyperparameters
alpha = 0.8       # learning rate
gamma = 0.95      # discount factor
epsilon = 0.1     # exploration rate
episodes = 1000
max_steps = 100

# Initialize Q-table
Q = np.zeros((state_size, action_size))

# Training loop
for episode in range(episodes):
    state, _ = env.reset()  # gymnasium returns a tuple (state, info)
    done = False

    for _ in range(max_steps):
        # Choose action (epsilon-greedy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)  # gymnasium returns 5 values
        done = terminated or truncated

        # Q-Update formula
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        if done:
            break

# Test the trained policy
print("\nTrained Q-Table:")
print(Q)

print("\nTesting trained policy:")
state, _ = env.reset()  # gymnasium returns a tuple (state, info)
env.render()

for _ in range(max_steps):
    action = np.argmax(Q[state, :])
    next_state, reward, terminated, truncated, _ = env.step(action)  # gymnasium returns 5 values
    done = terminated or truncated
    state = next_state
    env.render()
    if done:
        print(f"Game finished with reward: {reward}")
        break
