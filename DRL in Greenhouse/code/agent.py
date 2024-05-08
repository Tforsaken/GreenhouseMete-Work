import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)





class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return torch.sigmoid(self.fc(x))



class GreenhouseEnvironment:
    def __init__(self, external_intensity):
        self.external_intensity = external_intensity
        self.internal_intensity = np.zeros_like(self.external_intensity)
        self.light_on = 0
        self.curtain_closed = 0

    def step(self, action):
        light_action, curtain_action = action
        self.light_on = light_action
        self.curtain_closed = curtain_action


        light_intensity = 0
        if self.light_on > 0.5:
            light_intensity += 400
        external_light = self.external_intensity * (1 - self.curtain_closed)
        self.internal_intensity = external_light + light_intensity


        reward = -abs(475 - self.internal_intensity)
        return reward



def train_model():
    external_intensity = generate_external_light_intensity()
    env = GreenhouseEnvironment(external_intensity)
    policy_net = PolicyNetwork().cuda()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    epochs = 50
    for epoch in range(epochs):
        state = torch.tensor([env.external_intensity[0]], dtype=torch.float32).cuda().unsqueeze(0)
        for t in range(len(env.external_intensity)):
            probs = policy_net(state)
            action = (probs > 0.5).float().cpu().data.numpy()[0]
            loss = -torch.log(probs + 1e-5) * reward
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            state = torch.tensor([env.external_intensity[t]], dtype=torch.float32).cuda().unsqueeze(0)

    return external_intensity, env.internal_intensity, policy_net



external_intensity, internal_intensity, policy_net = train_model()

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(external_intensity, label='External Light Intensity')
plt.ylabel('Intensity (umol/m2/s)')
plt.title('External Light Intensity Over Time')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(internal_intensity, label='Internal Light Intensity')
plt.ylabel('Intensity (umol/m2/s)')
plt.title('Internal Light Intensity Over Time')
plt.legend()


plt.subplot(3, 1, 3)
time_points = np.linspace(0, 24, len(internal_intensity))
light_status = [1 if intensity >= 450 else 0 for intensity in internal_intensity]
curtain_status = [1 if intensity <= 475 else 0 for intensity in internal_intensity]
plt.step(time_points, light_status, label='Light ON', where='post')
plt.step(time_points, curtain_status, label='Curtain Closed', where='post')
plt.xlabel('Time (hours)')
plt.ylabel('Status (ON=1/OFF=0)')
plt.title('Device Status Over Time')
plt.legend()

plt.tight_layout()
plt.show()