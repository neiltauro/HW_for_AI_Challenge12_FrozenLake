import gymnasium as gym
import numpy as np
import random
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time

# Generate a random 10x10 Frozen Lake map with more holes
random_map = generate_random_map(size=10, p=0.6)  # Lower p value means more holes (less safe tiles)

# Set up environment with the random map
env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=False)  # deterministic version

# Parameters
num_episodes = 2000
max_steps = 100
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1  # exploration rate

# Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Timing variables
update_q_value_time = 0

# Update Q-value function
def update_q_value(q_table, state, action, reward, new_state, learning_rate, discount_factor):
    global update_q_value_time
    start_time = time.time()

    old_value = q_table[state, action]
    next_max = np.max(q_table[new_state])
    new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
    q_table[state, action] = new_value

    update_q_value_time += time.time() - start_time


# Training loop
def train_q_learning():
    for episode in range(num_episodes):
        state, _ = env.reset()  # Gymnasium's reset() returns (state, info)
        for step in range(max_steps):
            # Choose action: epsilon-greedy
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            new_state, reward, done, truncated, _ = env.step(action)  # Gymnasium's step() returns (state, reward, done, truncated, info)

            update_q_value(q_table, state, action, reward, new_state, learning_rate, discount_factor)

            state = new_state
            if done or truncated:
                break


# Test the trained policy
def test_q_learning():
    print("\nTrained Q-Table:")
    print(q_table)

    print("\nTesting trained policy:")
    state, _ = env.reset()  # Gymnasium's reset() returns (state, info)
    env.render()

    for step in range(max_steps):
        action = np.argmax(q_table[state])
        new_state, reward, done, truncated, _ = env.step(action)  # Gymnasium's step() returns (state, reward, done, truncated, info)
        state = new_state
        env.render()
        if done or truncated:
            print(f"Game finished with reward: {reward}")
            break


if __name__ == "__main__":
    # Train the Q-learning agent
    train_q_learning()

    # Test the trained policy
    test_q_learning()
