hyperparams = {
    'seed':                101,     # random seed
    'buffer_size':         100000,  # size of the experience replay buffer
    'batch_size':          64,      # number of experiences to sample at each learning step
    'start_since':         64,      # number of experiences to store before it begins learning (must be >= 'batch_size')
    'gamma':               0.99,    # discount factor
    'target_update_every': 1,       # how often to update the target network
    'tau':                 1e-3,    # how much to update the target network at every update
    'lr':                  1e-3,    # learning rate
    'update_every':        1,       # how often to update the online network
    'priority_eps':        1e-3,    # small values added to priorities in order to have nonzero priorities
    'a':                   0.5,     # priority exponent parameter
    'n_multisteps':        3,       # number of steps to consider for multistep learning
    'v_min':               -50,     # minimum support value for distributional learning
    'v_max':               50,      # maximum support value for distributional learning
    'n_atoms':             51,      # number of supports for distributional learning
    'initial_sigma':       0.50,    # initial noise parameter value for noisy net
    'linear_type':         'noisy', # which linear layers to use ('linear' or 'noisy')
    'factorized':          True,    # whether to use factorized gaussian noise or not
    'clip':                None     # gradient clipping
}
