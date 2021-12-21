import os
import pygame
import sys
import time

import gym
import numpy as np
from gym import spaces
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
from stable_baselines.deepq.policies import MlpPolicy

SEED = 421

np.random.seed(SEED)

# Игровые настройки
SIZE = 10  # field size
START = (0, 0)  # Стартовая позиция тигра
ENEMIES = {(3, 5), (4, 5), (5, 5), (6, 5)}
RABBITS = {(4, 6), (9, 1), (6, 3)}

# настройки отображения
STEP_PAUSE = 0.2  # Пауза между ходами.

CELL_SIZE = 34
BORDER_SIZE = 2
FONT_SIZE = 12
STATUS_SIZE = FONT_SIZE * 2 + 5
WHITE = (255, 255, 255)
BLACK = (11, 74, 32)
BLUE = (0, 0, 200)
RED = (200, 0, 0)

N_DISCRETE_ACTIONS = 4  # up down left right
HEIGHT = 10
WIDTH = 10
VOID = 0
TIGER = 1
RABBIT = 2

# настройка pygame для отображения обучения.
pygame.init()

size = width, height = SIZE * CELL_SIZE + (SIZE - 1) * BORDER_SIZE, SIZE * CELL_SIZE + (
            SIZE - 1) * BORDER_SIZE + STATUS_SIZE

screen = pygame.display.set_mode(size)
sp_tiger = pygame.image.load("sprites/tig.png")
sp_rihn = pygame.image.load("sprites/rihn.png")
sp_step = pygame.image.load("sprites/step.png")
sp_grass = pygame.image.load("sprites/grass.png")
sp_rab = pygame.image.load("sprites/rab.png")
sp_start = pygame.image.load("sprites/start.png")


def pygame_render(env):
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit(0)

    print(env.tiger)

    screen.fill(BLACK)
    draw_grid()
    draw_grass()

    for step in env.steps:  # Рисуем путь.
        draw(sp_step, step)
    draw(sp_start, START)
    draw(sp_tiger, env.tiger)  # Рисуем тигра
    for rabbit in env.rabbits:  # Рисуем кроликов.
        draw(sp_rab, rabbit)
    for enemy in env.enemies:  # Рисуем врагов.
        draw(sp_rihn, enemy)

    pygame.display.flip()


def draw(sprite, pos):
    sprite_pos = (pos[1] * (CELL_SIZE + BORDER_SIZE), pos[0] * (CELL_SIZE + BORDER_SIZE))
    screen.blit(sprite, sprite_pos)


def draw_grid():
    # drawing cell borders
    i = 1
    while (i < SIZE):
        pygame.draw.line(screen, BLACK, (i * CELL_SIZE + (i - 1) * BORDER_SIZE + 1, 0),
                         (i * CELL_SIZE + (i - 1) * BORDER_SIZE + 1, SIZE * CELL_SIZE + (SIZE - 1) * BORDER_SIZE - 1),
                         BORDER_SIZE)
        i += 1
    i = 1
    while (i < SIZE):
        pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE + (i - 1) * BORDER_SIZE + 1),
                         (SIZE * CELL_SIZE + (SIZE - 1) * BORDER_SIZE - 1, i * CELL_SIZE + (i - 1) * BORDER_SIZE + 1),
                         BORDER_SIZE)
        i += 1


def draw_grass():
    i = 0
    while (i < SIZE):
        j = 0
        while (j < SIZE):
            screen.blit(sp_grass, (j * (CELL_SIZE + BORDER_SIZE), i * (CELL_SIZE + BORDER_SIZE)))
            j += 1
        i += 1


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        # 0 = void 1 = tiger 2 = rabbit
        self.observation_space = spaces.Box(low=0, high=2, shape=(HEIGHT * WIDTH,), dtype=int)
        print(self.observation_space)

        self.reset()

    def step(self, action):
        self.steps.append(self.tiger)  # Сохраняем истрию перемещений.

        action = self._get_action(action)

        if self._validate_action(action):
            self._eval_action(action)

        self.done = self._is_done()
        self.rabbit_find = self._is_rabbit_find()

        return np.concatenate(self.map).ravel(), self._get_reward(), self.done, {}

    def _get_action(self, action):
        t = self.tiger
        d = self._get_move_delta(action)
        return (t[0] + d[0], t[1] + d[1])

    def _get_move_delta(self, action):
        if action == 0:
            return (1, 0)
        if action == 1:
            return (0, 1)
        if action == 2:
            return (-1, 0)
        if action == 3:
            return (0, -1)

    def _validate_action(self, pos):
        self._action_valid = not self._out_border(*pos)
        return self._action_valid

    def _eval_action(self, pos):
        self.map[self.tiger] = 0
        self.tiger = pos
        self.map[self.tiger] = TIGER

    def _out_border(self, x, y):
        if x < 0 or y < 0 or x >= WIDTH or y >= HEIGHT:
            return True
        return False

    def _is_rabbit_find(self):
        return self._near_rabbit() <= 3

    def _near_rabbit(self):
        return min(map(lambda pos: self._get_len(pos, self.tiger), self.rabbits))

    def _get_len(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_reward(self):
        if self._action_valid == False:
            return -100
        if self.tiger in self.enemies:
            return -100
        if self.done:
            return 1000
        if self.try_catch:
            return 500
        return self._near_rabbit() * -1

    def _is_done(self):
        if not (self.tiger in self.rabbits):
            self.try_catch = False
            return False
        self.try_catch = True
        # тигр поймал одного из кроликов
        if np.random.random() < self.tiger_proba:
            return True

        # тигр не сумел поймать кролика, и кролик сбежал.
        self.tiger_proba += 0.1

        rabbit_pos = self.tiger
        rabbit_pos_index = self.rabbits.index(rabbit_pos)
        delta = (0, 0)
        m = 5
        ds = [(m, 0), (-m, 0), (0, m), (0, -m)]
        for (x, y) in ds:
            if not self._out_border(x + rabbit_pos[0], y + rabbit_pos[1]):
                delta = (x, y)
                break

        rabbit_pos = (rabbit_pos[0] + delta[0], rabbit_pos[1] + delta[1])

        self.rabbits[rabbit_pos_index] = rabbit_pos
        self.map[rabbit_pos] = RABBIT

        return False

    def reset(self):
        self.steps = []
        self.try_catch = False
        self.rabbit_find = False
        self.tiger = START
        self.tiger_proba = 0.5

        self.rabbits = list(RABBITS)
        self.enemies = list(ENEMIES)

        self.map = np.zeros((WIDTH, HEIGHT), dtype=int)
        self.map[self.tiger] = TIGER
        for x, y in self.rabbits:
            self.map[x, y] = RABBIT

        return np.concatenate(self.map).ravel()

    def render(self, mode='human', close=False):
        print(self.map)
        pygame_render(self)


env = CustomEnv()
# проверка что данная среда подходит для обучения.
check_env(env, warn=True)

ans = input("1. Load model or 2. learning new model?")

if ans == "1" or ans == "":
    model = DQN.load("good_model")
else:
    model = DQN(MlpPolicy, env, verbose=1, seed=SEED)
    timestemps = int(input("learning timestemps (in sec, recomended 20)?")) * 1000
    # Процесс обучения, 20000
    model.learn(total_timesteps=timestemps)

actions = list(range(env.action_space.n))
reward = 0
report = []

# report_count = int(input("Report count?"))
report_count = 10

for i in range(report_count):
    obs = env.reset()
    dones = False
    step = 0
    while not dones and step < 50:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

        print(i, action)
        step += 1
        time.sleep(STEP_PAUSE)

    s = str(i) + ": done:" + str(dones) + " steps:" + str(step) + " tiger proba:" + str(env.tiger_proba)
    report.append(s)
    print(s)

for s in report:
    print(s)

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit(0)
