#!/usr/bin/env bash

# for i in {1..8}
# do
#     tmux new-session -s $i -d "conda activate RL; python custom_parse_start.py --device_num=0 --save_num=$i --risk_aversion=$i --n_episodes=5000; read"
# done

# for i in {11..16}
# do
#     tmux new-session -s $i -d "conda activate RL; python custom_parse_start.py --device_num=1 --save_num=$i --risk_aversion=$i --n_episodes=5000; read"
# done

# for i in {17..23}
# do
#     tmux new-session -s $i -d "conda activate RL; python custom_parse_start.py --device_num=2 --save_num=$i --risk_aversion=$i --n_episodes=5000; read"
# done

# for i in {24..30}
# do
#     tmux new-session -s $i -d "conda activate RL; python custom_parse_start.py --device_num=3 --save_num=$i --risk_aversion=$i --n_episodes=5000; read"
# done

start_from=1
num_workers=30

quarter=$((num_workers/4.0))

for ((i=$start_from;i<$((start_from+num_workers));i++))
do
    tmux new-session -s $i -d "/home/jeffrey/anaconda3/envs/RL/bin/python custom_parse_start.py --device_num=${$(($((i-start_from))/quarter))%.*} --save_num=$i --risk_aversion=$i --n_episodes=5000; read"
done