# Memory Buffer & Agent Hyperparameters
SEED = 101              # seed for random number generation
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
START_SINCE = int(8e3)  # number of steps to collect before start learning
GAMMA = 0.99            # discount factor
T_UPDATE = 4            # how often to update the target network
TAU = 1e-3              # for soft update of target parameters
LR = 3e-4               # learning rate
WEIGHT_DECAY = 0        # Weight decay value
UPDATE_EVERY = 4        # how often to update the online network
A = 0.5                 # randomness vs priority parameter
INIT_BETA = 0.4         # initial importance-sampling weight
P_EPS = 1e-3            # priority epsilon
N_STEPS = 3             # multi-step number of steps
V_MIN = -50             # distributional learning maximum support bound
V_MAX = 50              # distributional learning minimum support bound
CLIP = None             # gradient clipping (`None` to disable)

# Model Hyperparameters
N_ATOMS = 51            # number of atoms used in distributional network
INIT_SIGMA = 0.5        # initial noise parameter values
LINEAR = 'noisy'        # type of linear layer ('linear', 'noisy')
FACTORIZED = True       # whether to use factorized gaussian noise


# Default Checks
assert isinstance(SEED, int), "invalid default SEED"
assert isinstance(BUFFER_SIZE, int) and BUFFER_SIZE > 0, "invalid default BUFFER_SIZE"
assert isinstance(BATCH_SIZE, int) and BATCH_SIZE > 0, "invalid default BATCH_SIZE"
assert isinstance(START_SINCE, int) and START_SINCE >= BATCH_SIZE, "invalid default START_SINCE"
assert isinstance(GAMMA, (int, float)) and 0 <= GAMMA <= 1, "invalid default GAMMA"
assert isinstance(T_UPDATE, int) and T_UPDATE > 0, "invalid default T_UPDATE"
assert isinstance(TAU, (int, float)) and 0 <= TAU <= 1, "invalid default TAU"
assert isinstance(LR, (int, float)) and LR >= 0, "invalid default LR"
assert isinstance(WEIGHT_DECAY, (int, float)) and WEIGHT_DECAY >= 0, "invalid default WEIGHT_DECAY"
assert isinstance(UPDATE_EVERY, int) and UPDATE_EVERY > 0, "invalid default UPDATE_EVERY"
assert isinstance(A, (int, float)) and 0 <= A <= 1, "invalid default A"
assert isinstance(INIT_BETA, (int, float)) and 0 <= INIT_BETA <= 1, "invalid default INIT_BETA"
assert isinstance(P_EPS, (int, float)) and P_EPS >= 0, "invalid default P_EPS"
assert isinstance(N_STEPS, int) and N_STEPS > 0, "invalid default N_STEPS"
assert isinstance(V_MIN, (int, float)) and isinstance(V_MAX, (int, float)) and V_MIN < V_MAX, "invalid default V_MIN"
if CLIP: assert isinstance(CLIP, (int, float)) and CLIP >= 0, "invalid default CLIP"
assert isinstance(N_ATOMS, int) and N_ATOMS > 0, "invalid default N_ATOMS"
assert isinstance(INIT_SIGMA, (int, float)), "invalid default INIT_SIGMA"
assert isinstance(LINEAR, str) and LINEAR.lower() in ('linear', 'noisy'), "invalid default LINEAR"
assert isinstance(FACTORIZED, bool), "invalid default FACTORIZED"
