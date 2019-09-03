import os
import numpy as np
env_name = ["bollinger","rsi", "macd", "stochastic"]
agents= np.arange(1,30)

for env in env_name:
    for agent_num in agents:
        location = os.path.join("saves", "transfer",env, str(agent_num),"TradingGym_Rainbow_400.pth")
        if not os.path.exists(location):
            # print(env, agent_num)
            cmd = f'tmux new-session -s {env}{agent_num} -d "/home/jeffrey/anaconda3/envs/RL/bin/python -i transfer_learning.py --device_num={agent_num % 3} --save_num={agent_num} --risk_aversion={agent_num} --n_episodes=502 --environment={env}; read"'
            # print(location)
            os.system(cmd)
            # print(cmd)