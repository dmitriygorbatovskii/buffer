import env_2
import numpy as np
import matplotlib.pyplot as plt

import math, random
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from collections import deque



env = env_2.test_env()

model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 4)
)

optimizer = optim.Adam(model.parameters())

replay_buffer = deque(maxlen=1000)

num_frames = 1000
batch_size = 32
gamma = 0.99

losses = []
all_rewards = []
episode_reward = 0
a = []
b = []

state = env.reset()
min_steps = env.min_steps()
steps = 0
for frame_idx in range(1, num_frames + 1):
    epsilon = 0.0

    if random.random() > epsilon:
        st = torch.FloatTensor(state).unsqueeze(0)
        q_value = model(st)
        action = q_value.max(1)[1].data[0]
    else:
        action = random.randrange(4)

    next_state, reward, done, _ = env.step(int(action))

    st = np.expand_dims(state, 0)
    next_st = np.expand_dims(next_state, 0)
    replay_buffer.append((st, action, reward, next_st, done))

    state = next_state
    episode_reward += reward
    steps += 1

    if not frame_idx % 100:
        print(frame_idx)

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        b.append(steps - min_steps)
        a.append(frame_idx)
        episode_reward = 0
        min_steps = env.min_steps()
        steps = 0

    if len(replay_buffer) > batch_size:
        st, action, rew, next_st, done = zip(*random.sample(replay_buffer, batch_size))
        st = np.concatenate(st)
        next_st = np.concatenate(next_st)

        st = torch.FloatTensor(np.float32(st))
        next_st = torch.FloatTensor(np.float32(next_st))
        action = torch.LongTensor(action)
        rew = torch.FloatTensor(rew)
        done = torch.FloatTensor(done)

        q_values = model(st)
        next_q_values = model(next_st)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rew + gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.data).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data)

plt.plot(a, b)
plt.show()
plt.plot(a, all_rewards)
plt.show()