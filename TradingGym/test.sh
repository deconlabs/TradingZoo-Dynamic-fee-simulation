for i in {10..12}
do
    tmux new-session -s $i -d "python remove.py --i=$i"
done