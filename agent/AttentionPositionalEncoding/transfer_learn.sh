env=bollinger
for i in 0 5 10 15 20 25 30
do
    tmux new-session -s $env-$i -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$(($i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --environment=macd; read"
done

env=rsi
for i in 0 5 10 15 20 25 30
do
    tmux new-session -s $env-$i -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$(($i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --environment=macd; read"
done

env=stochastic
for i in 0 5 10 15 20 25 30
do
    tmux new-session -s $env-$i -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$(($i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --environment=macd; read"
done

# env=macd
# for i in 0 5 10 15 20 25 30
# do
#     tmux kill-session -t transfer-$i
# done


# fee = .005
# for i in 0 5 10 15 20 25 30
# do
#     tmux new-session -s $i-005 -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --fee=.005; read"
# done

# # fee = .003
# for i in 0 5 10 15 20 25 30
# do
#     tmux new-session -s $i-003 -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --fee=.003; read"
# done

# # fee = 0.000
# for i in 0 5 10 15 20 25 30
# do
#     tmux new-session -s $i-000 -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --fee=.000; read"
# done