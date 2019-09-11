# #fee = .0
# for i in 0 5 10 15 20 25 
# do
#     tmux new-session -s $i-0 -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=502 --fee=.0; read"
# done

# # fee = .005
# for i in 0 5 10 15 20 25 
# do
#     tmux new-session -s $i-005 -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=502 --fee=.005; read"
# done

# # # fee = .003
# for i in 0 5 10 15 20 25 
# do
#     tmux new-session -s $i-003 -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=502 --fee=.003; read"
# done

# basic
# for ((i=$start_from;i<$((start_from+num_workers));i++))
# do
#     tmux new-session -s $i-001 -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --fee=.001; read"
# done
 
env=macd 
for i in 0 5 10 15 20 25 
do
    tmux new-session -s $i-$env -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=502 --environment=macd; read"
done

env=rsi
for i in 0 5 10 15 20 25 
do
    tmux new-session -s $i-$env -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=502 --environment=rsi; read"
done