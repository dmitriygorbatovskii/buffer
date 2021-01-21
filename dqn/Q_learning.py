import env_2
import numpy as np
import matplotlib.pyplot as plt


# инициализируем состояния и награды
Q = []
for y in range(-9, 10):
    for x in range(-14, 15):
        Q.append([y, x, 0, 0, 0, 0])

y = []
x = []
alpha = 0.01 # шаг обучения
gma = 1 # discount factor

env = env_2.test_env()
for i in range(1000):
    state = env.reset()
    min_steps = env.min_steps() # минимальное количество шагов для достижения цели
    v = 0
    while True:
        current_step = [item for item in Q if item[:2] == state][0]
        action = current_step[2:].index(max(current_step[2:]))
        next_state, reward, done, info = env.step(action)

        next_step = [item for item in Q if item[:2] == next_state][0]
        best_next_action = next_step[2:].index(max(next_step[2:]))

        current_step[action+2] += alpha * (reward + gma * (next_step[best_next_action+2]) - current_step[action+2])

        state = next_state
        v += 1
        if done:
            y.append(v-min_steps)
            x.append(i)
            break

# y - количество шагов в эпизоде - количство минимально необходимых
# x - эпохи
poly = np.polyfit(x, y, 5)
poly_y = np.poly1d(poly)(x)
plt.plot(x, y)
plt.plot(x, poly_y)
plt.show()
