start_from=1
num_workers=30

#fee = .005
for ((i=$start_from;i<$((start_from+num_workers));i++))
do
    tmux new-session -s $i-005 -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=402 --fee=.005; read"
done

# fee = .003
for ((i=$start_from;i<$((start_from+num_workers));i++))
do
    tmux new-session -s $i-003 -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=402 --fee=.003; read"
done

# basic
# for ((i=$start_from;i<$((start_from+num_workers));i++))
# do
#     tmux new-session -s $i-001 -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --fee=.001; read"
# done


# 파생상품 환경으로 
# env=bollinger
# for ((i=$start_from;i<$((start_from+num_workers));i++))
# do
#     tmux new-session -s $i$env -d "/home/jeffrey/anaconda3/envs/RL/bin/python -i transfer_learning.py --device_num=$(($i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --environment=bollinger; read"
# done

# env=rsi
# for ((i=$start_from;i<$((start_from+num_workers));i++))
# do
#     tmux new-session -s $i$env -d "/home/jeffrey/anaconda3/envs/RL/bin/python -i transfer_learning.py --device_num=$(($i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --environment=rsi; read"
# done

# env=macd
# for ((i=$start_from;i<$((start_from+num_workers));i++))
# do
#     tmux new-session -s $i$env -d "/home/jeffrey/anaconda3/envs/RL/bin/python -i transfer_learning.py --device_num=$(($i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --environment=macd; read"
# done


# env=stochastic
# for ((i=$start_from;i<$((start_from+num_workers));i++))
# do
#     tmux new-session -s $i$env -d "/home/jeffrey/anaconda3/envs/RL/bin/python -i transfer_learning.py --device_num=$(($i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --environment=stochastic; read"
# done

# #kill
# env=rsi
# for i in {1..30}
# do
#     tmux kill-session -t $i$env
# done