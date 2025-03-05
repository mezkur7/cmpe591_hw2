import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from homework2 import Hw2Env

# Hyperparameters
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_DECAY_ITER = 5
EPSILON_MIN = 0.05
LEARNING_RATE = 0.001
BATCH_SIZE = 64
UPDATE_FREQ = 10
TARGET_NETWORK_UPDATE_FREQ = 50
BUFFER_LENGTH = 100000
N_ACTIONS = 8
NUM_EPISODES = 10000

# Define the DQN Model


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Initialize environment and networks
env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
policy_net = DQN(input_dim=6, output_dim=N_ACTIONS).to(torch.device("cpu"))
target_net = DQN(input_dim=6, output_dim=N_ACTIONS).to(torch.device("cpu"))
target_net.load_state_dict(policy_net.state_dict())
torch.set_num_threads(torch.get_num_threads() * 4)

optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE)
policy_net = torch.jit.script(policy_net)
replay_buffer = deque(maxlen=BUFFER_LENGTH)


def select_action(state, epsilon):
    if random.random() < epsilon:
        return np.random.randint(N_ACTIONS)  # Explore
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(policy_net(state_tensor)).item()


def train_dqn():
    if len(replay_buffer) < BATCH_SIZE:
        return

    indices = np.random.choice(len(replay_buffer), BATCH_SIZE, replace=False)
    batch = np.array([replay_buffer[i] for i in indices], dtype=object)

    states = torch.tensor(np.vstack(batch[:, 0]), dtype=torch.float32)
    actions = torch.tensor(np.array(batch[:, 1], dtype=np.int64), dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(np.array(batch[:, 2], dtype=np.int64), dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.vstack(batch[:, 3]), dtype=torch.float32)
    dones = torch.tensor(np.array(batch[:, 4], dtype=np.int64), dtype=torch.float32).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1, keepdim=True)[0].detach()
    target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

    loss = nn.SmoothL1Loss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


""" 
# Training Loop
epsilon = EPSILON
epsilon_decay_counter = 0
rewards_per_episode = []
rps_per_episode = []

for episode in range(NUM_EPISODES):
    env.reset()
    state = env.high_level_state()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, is_terminal, is_truncated = env.step(action)
        next_state = env.high_level_state()
        done = is_terminal or is_truncated
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        steps += 1

        if steps % UPDATE_FREQ == 0:
            train_dqn()

        if epsilon_decay_counter % EPSILON_DECAY_ITER == 0:
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        epsilon_decay_counter += 1

    if episode % TARGET_NETWORK_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    rewards_per_episode.append(total_reward)
    rps_per_episode.append(total_reward / steps)
    print(f"Episode {episode}: Reward={total_reward:.2f}, RPS={total_reward/steps:.4f}")

# Save results
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode")
plt.show()


plt.plot(rps_per_episode)
plt.xlabel("Episode")
plt.ylabel("Reward per Step (RPS)")
plt.title("RPS per Episode")
plt.show()

np.array(rewards_per_episode).tofile("rewards_per_episode_10000.npy")
np.array(rps_per_episode).tofile("rps_per_episode_10000.npy")
torch.save(policy_net.state_dict(), "policy_net_10000.pth")
torch.save(target_net.state_dict(), "target_net_10000.pth")
torch.save(optimizer.state_dict(), "optimizer_10000.pth") 
"""


def train():
    epsilon = EPSILON
    epsilon_decay_counter = 0
    rewards_per_episode = []
    rps_per_episode = []

    for episode in range(NUM_EPISODES):
        env.reset()
        state = env.high_level_state()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = select_action(state, epsilon)
            next_state, reward, is_terminal, is_truncated = env.step(action)
            next_state = env.high_level_state()
            done = is_terminal or is_truncated
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1

            if steps % UPDATE_FREQ == 0:
                train_dqn()

            if epsilon_decay_counter % EPSILON_DECAY_ITER == 0:
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            epsilon_decay_counter += 1

        if episode % TARGET_NETWORK_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_per_episode.append(total_reward)
        rps_per_episode.append(total_reward / steps)
        print(f"Episode {episode}: Reward={total_reward:.2f}, RPS={total_reward/steps:.4f}")

    # Save results
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode")
    plt.show()

    plt.plot(rps_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Reward per Step (RPS)")
    plt.title("RPS per Episode")
    plt.show()

    np.array(rewards_per_episode).tofile("rewards_per_episode_10000.npy")
    np.array(rps_per_episode).tofile("rps_per_episode_10000.npy")
    torch.save(policy_net.state_dict(), "policy_net_10000.pth")
    torch.save(target_net.state_dict(), "target_net_10000.pth")
    torch.save(optimizer.state_dict(), "optimizer_10000.pth")


def load_models(policy_path, target_path):
    """Loads trained policy and target models."""
    policy_net = DQN(input_dim=6, output_dim=N_ACTIONS).to(torch.device("cpu"))
    target_net = DQN(input_dim=6, output_dim=N_ACTIONS).to(torch.device("cpu"))
    policy_net.load_state_dict(torch.load("policy_net_10000.pth"))
    target_net.load_state_dict(torch.load("target_net_10000.pth"))
    policy_net.eval()
    target_net.eval()
    return policy_net, target_net


def test_dqn(policy_net, n_episodes=50, n_actions=8, render=False):
    """Runs the trained DQN model and evaluates performance."""
    env = Hw2Env(n_actions=n_actions, render_mode="gui" if render else "offscreen")
    total_rewards = []
    rps_values = []

    for episode in range(n_episodes):
        env.reset()
        state = env.high_level_state()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(policy_net(state_tensor)).item()

            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            total_reward += reward
            steps += 1
            state = next_state

        total_rewards.append(total_reward)
        rps_values.append(total_reward / steps if steps > 0 else 0)
        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, RPS={total_reward/steps:.4f}")

    env.close()
    return total_rewards, rps_values

# Plot results


def plot_test_results(total_rewards, rps_values):
    """Plots the reward and RPS values for the test episodes."""
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards, label="Total Reward per Episode", color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Model Test: Reward per Episode")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(rps_values, label="Reward per Step (RPS)", color='red')
    plt.xlabel("Episode")
    plt.ylabel("RPS")
    plt.title("DQN Model Test: RPS per Episode")
    plt.legend()
    plt.show()


def test():
    policy_net, target_net = load_models("policy_net_10000.pth", "target_net_10000.pth")
    total_rewards, rps_values = test_dqn(policy_net, n_episodes=50, n_actions=8, render=False)
    plot_test_results(total_rewards, rps_values)
