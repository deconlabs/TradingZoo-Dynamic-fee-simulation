import os
import numpy as np
env_name = ["bollinger","rsi", "macd", "stochastic","0.000",'0.005','0.003']
agents= np.arange(1,30)

for env in env_name:
    for agent_num in agents:
        location = os.path.join("saves", "transfer",env, str(agent_num),"TradingGym_Rainbow_400.pth")
        # print(location)
        if not os.path.exists(location):
            print(env, agent_num)
            if env.startswith("0."):
                env = env[2:]
                cmd = f'tmux new-session -s {agent_num}-{env} -d "/home/jeffrey/anaconda3/envs/RL/bin/python -i transfer_learning.py --device_num={agent_num % 3} --save_num={agent_num} --risk_aversion={agent_num} --n_episodes=502 --environment={env}; read"'    
            else:
                cmd = f'tmux new-session -s {agent_num}-{env} -d "/home/jeffrey/anaconda3/envs/RL/bin/python -i transfer_learning.py --device_num={agent_num % 3} --save_num={agent_num} --risk_aversion={agent_num} --n_episodes=502 --environment={env}; read"'
            # print(location)
            os.system(cmd)
            # print(cmd)