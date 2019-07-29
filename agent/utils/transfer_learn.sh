start_from=1
num_workers=27


#fee = .005
# for ((i=$start_from;i<$((start_from+num_workers));i++))
# do
#     tmux new-session -s $i -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --fee=.005; read"
# done

# fee = .003
for ((i=$start_from;i<$((start_from+num_workers));i++))
do
    tmux new-session -s $i -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --fee=.003; read"
done

# basic
# for ((i=$start_from;i<$((start_from+num_workers));i++))
# do
#     tmux new-session -s $i -d "/home/jeffrey/anaconda3/envs/RL/bin/python transfer_learning.py --device_num=$((i%4)) --save_num=$i --risk_aversion=$i --n_episodes=500 --fee=.005; read"
# done
    