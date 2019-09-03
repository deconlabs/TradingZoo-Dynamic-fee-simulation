#!/usr/bin/env bash

start_from=1
num_workers=30

for ((i=$start_from;i<$((start_from+num_workers));i++))
do
    tmux new-session -s $i -d "/home/jeffrey/anaconda3/envs/RL/bin/python attention_start.py --device_num=$(($i%4)) --save_num=$i --risk_aversion=$i --n_episodes=1000; read"
done

# for ((i=$start_from;i<$((start_from+num_workers));i++))
# do
#     if [ $(($i%4)) -eq 0 ]
#     then
#         tmux new-session -s $i -d "/home/jeffrey/anaconda3/envs/RL/bin/python dqn_start.py --device_num=$(($i%4)) --save_num=$i --risk_aversion=$i --n_episodes=5000; read"
#     fi
    
# done