for i in {1..40}
do
    tmux kill-session -t $i
done

# tmux new-session -s "removepy" -d "conda activate RL; python remove.py; read"