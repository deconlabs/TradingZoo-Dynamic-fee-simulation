# for env in "bollinger" "rsi" "macd" "stochastic"
# do
#     for i in {1..40}
#     do
#         tmux kill-session -t $env$i
#     done
# done

# tmux new-session -s "removepy" -d "conda activate RL; python remove.py; read"

start_from=1
num_workers=30
for ((i=$start_from;i<$((start_from+num_workers));i++))
do
    tmux kill-session -t $i
    # tmux kill-session -t $i-003
    # tmux kill-session -t $i-0
done
