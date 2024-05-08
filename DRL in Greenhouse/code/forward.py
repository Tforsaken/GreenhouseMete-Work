import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random



def simulate_temperature(day_length, max_temp, night_temp):

    return np.linspace(night_temp, max_temp, day_length // 2).tolist() + \
        np.linspace(max_temp, night_temp, day_length // 2).tolist()


class GreenhouseEnv:
    def __init__(self, outdoor_temperatures, target_temperature=25.0, max_steps=1000, noise_scale=0.1):
        self.outdoor_temperatures = outdoor_temperatures
        self.target_temperature = target_temperature
        self.tem_fanwei = (20.0, 30.0)
        self.max_steps = max_steps
        self.noise_scale = noise_scale
        self.reset()

    def reset(self):
        self.current_temperature = np.random.uniform(*self.tem_fanwei)
        self.current_step = 0
        return self._normalize_state(self.current_temperature)

    def _normalize_state(self, temperature):

        return np.array([(temperature - 20.0) / 10.0])

    def step(self, action):
        self.current_step += 1


        outdoor_temperature = self.outdoor_temperatures[self.current_step % len(self.outdoor_temperatures)]
        temperature_difference = self.current_temperature - outdoor_temperature

        fan_effect = action * -0.5


        temperature_change_due_to_diff = -0.25 * temperature_difference
        temperature_change = temperature_change_due_to_diff + fan_effect


        noise = np.random.normal(scale=self.noise_scale)
        temperature_change += noise

        self.current_temperature += temperature_change

        if abs(self.current_temperature - self.target_temperature) <= 1.0:
            reward = 1.0
        else:
            reward = -1.0


        done = self.current_step >= self.max_steps
        return self._normalize_state(self.current_temperature), reward, done



class DQNAgent(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)



# def train_dqn(env, agent, episodes, learning_rate=1e-3, gamma=0.99, epsilon_decay=0.995, epsilon_min=0.01):
#     loss_fn = nn.MSELoss()
#     optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
#     replay_buffer = deque(maxlen=10000)
#     epsilon = 1.0
#
#     for episode in range(episodes):
#         state = env.reset()
#         total_reward = 0
#         done = False
#         while not done:
#             state_tensor = torch.FloatTensor([state])
#             q_values = agent(state_tensor).squeeze(0)
#             if random.random() < epsilon:
#                 action = random.randint(0, 4)
#             else:
#                 action = torch.argmax(q_values).item()
#
#             next_state, reward, done = env.step(action)
#
#
#             replay_buffer.append((state, action, reward, next_state, done))
#
#
#             if len(replay_buffer) >= 128:
#                 batch = random.sample(replay_buffer, 128)
#                 states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
#
#                 state_batch = torch.FloatTensor(states)
#                 action_batch = torch.LongTensor(actions)
#                 reward_batch = torch.FloatTensor(rewards)
#                 next_state_batch = torch.FloatTensor(next_states)
#                 done_batch = torch.BoolTensor(dones)
#
#
#                 current_q_values = agent(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
#
#
#                 non_final_mask = ~done_batch
#                 non_final_next_states = next_state_batch[non_final_mask]
#                 next_state_values = torch.zeros(batch_size)
#                 next_state_values[non_final_mask] = agent(non_final_next_states).max(1)[0].detach()
#
#                 expected_q_values = reward_batch + (gamma * next_state_values)
#
#                 loss = loss_fn(current_q_values, expected_q_values)
#
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#             state = next_state
#             total_reward += reward
#
#
#         if (episode + 1) % 10 == 0:
#             print(f'Episode: {episode + 1}, Total Reward: {total_reward}')
#
#
#         epsilon = max(epsilon_min, epsilon * epsilon_decay)
#
#     return agent
#
def train_dqn(env, agent, episodes, learning_rate=1e-3, gamma=0.99, epsilon_decay=0.995, epsilon_min=0.01):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    replay_buffer = deque(maxlen=10000)
    epsilon = 1.0
    history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor([state])
            q_values = agent(state_tensor).squeeze(0)
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)


            replay_buffer.append((state, action, reward, next_state, done))


            if len(replay_buffer) >= 128:
                batch = random.sample(replay_buffer, 128)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

                state_batch = torch.FloatTensor(states)
                action_batch = torch.LongTensor(actions)
                reward_batch = torch.FloatTensor(rewards)
                next_state_batch = torch.FloatTensor(next_states)
                done_batch = torch.BoolTensor(dones)


                current_q_values = agent(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)


                non_final_mask = ~done_batch
                non_final_next_states = next_state_batch[non_final_mask]
                next_state_values = torch.zeros(128)  # 用128替代未定义的变量'batch_size'
                next_state_values[non_final_mask] = agent(non_final_next_states).max(1)[0].detach()


                expected_q_values = reward_batch + (gamma * next_state_values)


                loss = loss_fn(current_q_values, expected_q_values)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward


        history.append(total_reward)


        if (episode + 1) % 10 == 0:
            print(f'Episode: {episode + 1}, Total Reward: {total_reward}')


        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return agent, history


outdoor_temps = simulate_temperature(day_length=24, max_temp=30, night_temp=20)
env = GreenhouseEnv(outdoor_temperatures=outdoor_temps)
input_size = 1
output_size = 5


agent = DQNAgent(input_size, output_size)

trained_agent = train_dqn(env, agent, episodes=100, learning_rate=1e-3, gamma=0.99)