import gymnasium as gym
import numpy as np
import random
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time

from wrapper.qvalue_wrapper import run_hw_accel, float_to_fixed, fixed_to_float

random_map = generate_random_map(size=10, p=0.6)
env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=False)

num_episodes = 2000
max_steps = 100
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1

q_table = np.zeros((env.observation_space.n, env.action_space.n))
update_q_value_time = 0

def update_q_value(q_table, state, action, reward, new_state, learning_rate, discount_factor):
    global update_q_value_time
    start_time = time.time()

    old_val = q_table[state, action]
    next_max = np.max(q_table[new_state])

    new_value = run_hw_accel(
        float_to_fixed(old_val),
        float_to_fixed(reward),
        float_to_fixed(next_max),
        float_to_fixed(learning_rate),
        float_to_fixed(discount_factor)
    )
    q_table[state, action] = fixed_to_float(new_value)

    update_q_value_time += time.time() - start_time

def train_q_learning():
    for episode in range(num_episodes):
        state, _ = env.reset()
        for step in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            new_state, reward, done, truncated, _ = env.step(action)
            update_q_value(q_table, state, action, reward, new_state, learning_rate, discount_factor)
            state = new_state
            if done or truncated:
                break

def test_q_learning():
    print("\\nTrained Q-Table:")
    print(q_table)

    print("\\nTesting trained policy:")
    state, _ = env.reset()
    env.render()

    for step in range(max_steps):
        action = np.argmax(q_table[state])
        new_state, reward, done, truncated, _ = env.step(action)
        state = new_state
        env.render()
        if done or truncated:
            print(f"Game finished with reward: {reward}")
            break

if __name__ == "__main__":
    train_q_learning()
    test_q_learning()