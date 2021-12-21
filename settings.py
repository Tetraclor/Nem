# learning parameters
ALPHA = 1.0 # learning rate
GAMMA = 0.95 # discount

SIZE = 10 # field size

# trap cells init
# ENEMIES = set ()
ENEMIES = {(3,5), (4,5), (5,5), (6,5)}

# epsilon
EPSILON_START = 1.0
EPSILON_FINISH = 0.01

# start pos
X_START = 0
Y_START = 0
# final pos
X_FINISH = SIZE - 1
Y_FINISH = SIZE - 1

ACTIONS = 8 # actions number
DIRS = ["W", "E", "D", "C", "X", "Z", "A", "Q"]

STEP_PENALTY = -1.0 # step penalty
FINISH_BONUS = 100.0 # bonus penalty

SEED = 12345678

# pygame settings
CELL_SIZE = 34
BORDER_SIZE = 2
FONT_SIZE = 12
STATUS_SIZE = FONT_SIZE*2+5
WHITE = (255,255,255)
BLACK = (11,74,32)
BLUE = (0,0,200)
RED = (200,0,0)

STEP_PAUSE = 0.5
LRN_PAUSE = 0.001
ATTEMPT_PAUSE = 0.1

NONE_FLAG = 0
WALL_FLAG = 1
ENEMY_FLAG = 2
BORDER_FLAG = 3
FINISH_FLAG = 4