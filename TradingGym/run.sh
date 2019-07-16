#!/usr/bin/env bash

for i in {1..15}
do
    tmux new-session -s $i -d "python custom_parse_start.py --device_num=0 --save_num=$i --risk_aversion=$i --n_episodes=5000; read"
done

for i in {16..30}
do
    tmux new-session -s $i -d "python custom_parse_start.py --device_num=1 --save_num=$i --risk_aversion=$i --n_episodes=5000; read"
done

for i in {31..45}
do
    tmux new-session -s $i -d "python custom_parse_start.py --device_num=2 --save_num=$i --risk_aversion=$i --n_episodes=5000; read"
done

for i in {46..60}
do
    tmux new-session -s $i -d "python custom_parse_start.py --device_num=3 --save_num=$i --risk_aversion=$i --n_episodes=5000; read"
done