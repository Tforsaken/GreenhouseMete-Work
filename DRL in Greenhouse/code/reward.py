import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class GreenhouseEnv:
    def __init__(self, outdoor_temperatures, target_temperature_range, tem_fanwei=(20, 35), max_steps=24,
                 noise_scale=0.1):
        self.outdoor_temperatures = outdoor_temperatures
        self.target_temperature_range = target_temperature_range
        self.tem_fanwei = tem_fanwei
        self.max_steps = max_steps
        self.noise_scale = noise_scale
        self.reset()

    def reset(self):
        self.current_temperature = np.random.uniform(*self.tem_fanwei)
        self.current_step = 0
        return self.current_temperature

    def step(self, fan_power, heater_power):
        global temperature_change_due_to_diff
        self.current_step += 1


        outdoor_temperature = self.outdoor_temperatures[self.current_step % len(self.outdoor_temperatures)]

        temperature_difference = self.current_temperature - outdoor_temperature
        # if  fan_power== 1:
        #     fan_effect = 0.5
        # if  fan_power== 2:
        #     fan_effect = 0.7
        # if  fan_power== 3:
        #     fan_effect = 0.84
        # if  fan_power== 4:
        #     fan_effect = 1
        # if  heater_power== 1:
        #     heater_effect = 0.5
        # if  heater_power== 2:
        #     heater_effect = 0.7
        # if  heater_power== 3:
        #     heater_effect = 0.84
        # if 4 == heater_power:
        #     heater_effect = 1
        #
        # # fan_effect = fan_power * 0.5
        # # heater_effect = heater_power * 0.5
        # if temperature_difference > 7:
        #     temperature_change_due_to_diff = -0.7 * temperature_difference
        # if temperature_difference < -7:
        #     temperature_change_due_to_diff = -0.7 * temperature_difference
        # if temperature_difference > 5 or temperature_difference < -5:
        #     temperature_change_due_to_diff = -0.6 * temperature_difference
        # if temperature_difference > 4 or temperature_difference < -4:
        #     temperature_change_due_to_diff = -0.55 * temperature_difference
        # if temperature_difference > 3 or temperature_difference < -3:
        #     temperature_change_due_to_diff = -0.5 * temperature_difference
        # if temperature_difference > 2 or temperature_difference < -2:
        #     temperature_change_due_to_diff = -0.45 * temperature_difference
        # if temperature_difference > 1 or temperature_difference < -1:
        #     temperature_change_due_to_diff = -0.1 * temperature_difference
        if fan_power == 0:
            fan_effect = 0
        elif fan_power == 1:
            fan_effect = 1.4
        elif fan_power == 2:
            fan_effect = 2
        elif fan_power == 3:
            fan_effect = 3

        if heater_power == 0:
            heater_effect = 0
        elif heater_power == 1:
            heater_effect = 1.4
        elif heater_power == 2:
            heater_effect = 2
        elif heater_power == 3:
            heater_effect = 3

        if temperature_difference > 5 or temperature_difference < -5:
            temperature_change_due_to_diff = -0.6 * temperature_difference
        elif temperature_difference > 4 or temperature_difference < -4:
            temperature_change_due_to_diff = -0.55 * temperature_difference
        elif temperature_difference > 3 or temperature_difference < -3:
            temperature_change_due_to_diff = -0.5 * temperature_difference
        elif temperature_difference > 2 or temperature_difference < -2:
            temperature_change_due_to_diff = -0.45 * temperature_difference
        elif temperature_difference > 1 or temperature_difference < -1:
            temperature_change_due_to_diff = -0.1 * temperature_difference
        else:
            temperature_change_due_to_diff = -0.7 * temperature_difference

        temperature_change = temperature_change_due_to_diff - fan_effect + heater_effect




        self.current_temperature += temperature_change
         self.current_temperature = np.clip(self.current_temperature, *self.target_temperature_range)


        if self.current_temperature > 29:
            # if fan_power == 0:
            #     reward = -0.1
            if fan_power == 1:
                reward = 0.1
            elif fan_power == 2:
                reward = 0.2
            elif fan_power == 3:
                reward = 1
            else:
                reward = -0.1
        # elif self.current_temperature > 27:
        #     if fan_power == 1:
        #         reward = 0.3
        #     elif fan_power == 2:
        #         reward = 1
        #     elif fan_power == 3:
        #         reward = 0.3
        #     else :
        #         reward = -0.1

        elif self.current_temperature < 21:

            if heater_power == 1:
                reward = 0.3
            elif heater_power == 2:
                reward = 0.3
            elif heater_power == 3:
                reward = 1
            else:
                reward = -0.1
        # elif self.current_temperature <23:
        #     if heater_power == 1:
        #         reward = 0.3
        #     elif heater_power == 2:
        #         reward = 1
        #     elif heater_power == 3:
        #         reward = 0.3
        #     else:
        #         reward = -0.1
        #
        elif self.target_temperature_range[0] <= self.current_temperature <= self.target_temperature_range[1]:
            reward = 2
        else:
            reward = 0.1
        # # elif self.target_temperature_range[0]+1 <= self.current_temperature <= self.target_temperature_range[1]-1:
        # #     reward = 0.2
        # # elif self.target_temperature_range[0] + 2 <= self.current_temperature <= self.target_temperature_range[1] - 2:
        # #     reward = -0.1
        # else:
        #     reward = -0.5


        done = self.current_step >= self.max_steps

        return self.current_temperature, reward, done


class DQNAgent(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNAgent, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class QLearning:
    def __init__(self, num_states, num_actions, learning_rate=0.01, gamma=0.9, epsilon=0.09):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))

    def update_q_table(self, state, action, reward, next_state):
        state_index = int(round(state))
        next_state_index = int(round(next_state))
        max_next_q_value = np.max(self.q_table[next_state_index])
        new_q_value = reward + self.gamma * max_next_q_value
        self.q_table[state_index, action] += self.learning_rate * (new_q_value - self.q_table[state_index, action])




def inference_with_greenhouse(agent, outdoor_temperatures):
    # Create greenhouse environment
    env = GreenhouseEnv(outdoor_temperatures, target_temperature_range)

    # Initialize environment state
    current_temperature = outdoor_temperatures[0]
    current_outdoor_temperature_index = 0
    actions = []

    # Store temperatures and rewards
    indoor_temperatures = [current_temperature]
    outdoor_temperatures_used = [outdoor_temperatures[0]]
    rewards = []

    # Inference process
    while True:
        # Choose action
        #  state = np.array([current_temperature, outdoor_temperatures[current_outdoor_temperature_index]])
        state = env.current_temperature
        outdoor_temperature = env.outdoor_temperatures[env.current_step % len(env.outdoor_temperatures)]

        # 根据当前温度、外部温度和温度差选择动作
        action_input = torch.tensor([[state, outdoor_temperature]]).float()  # 更新这里
        action = agent(action_input).argmax().item()
        #
        # state = np.array([current_temperature, outdoor_temperatures[current_outdoor_temperature_index]])
        # q_values = agent(torch.FloatTensor(state))
        # action = torch.argmax(q_values).item()
        actions.append(action)

        # Parse action into fan and heater power
        if action > 3:
            fan_power = 0
            heater_power = action - 4
        else:
            fan_power = action
            heater_power = 0

        next_temperature, reward, done = env.step(fan_power, heater_power)

        # Store data
        indoor_temperatures.append(next_temperature)
        outdoor_temperatures_used.append(outdoor_temperatures[current_outdoor_temperature_index])
        rewards.append(reward)

        # Update state
        current_temperature = next_temperature
        current_outdoor_temperature_index = (current_outdoor_temperature_index + 1) % len(outdoor_temperatures)

        if done:
            break

    return actions, indoor_temperatures, outdoor_temperatures_used, rewards



