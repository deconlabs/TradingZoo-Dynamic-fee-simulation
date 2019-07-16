for i in {1..60}
do
    tmux kill-session -t $i
done