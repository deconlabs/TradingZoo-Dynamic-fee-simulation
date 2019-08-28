for i in {1..40}
do
    tmux kill-session -t fee3_$i
done

# tmux new-session -s "removepy" -d "conda activate RL; python remove.py; read"

# start_from=1
# num_workers=23
# for env in bollinger default macd rsi stochastic volume
# do
#     for ((i=$start_from;i<$((start_from+num_workers));i++))
#     do
#         tmux kill-session -t $i$env
#     done
# done