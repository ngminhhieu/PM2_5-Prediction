#!/bin/bash

git clone https://github.com/ngminhhieu/PM2_5-Prediction
while [ ! -d "PM2_5-Prediction" ]
do
  sleep 2 # or less like 0.2
done
cd "PM2_5-Prediction"
tmux new-session -d -s real
POP_WINDOW=( "pop_40" "pop_50" "pop_60" "pop_70" )
index=(0 1 2 3)
population=(40 50 60 70)
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux new-window -n ${POP_WINDOW[$i]} " ENTER
done

for i in "${index[@]}"
do
    tmux send-keys -t real "tmux send-keys -t ${POP_WINDOW[$i]} 'python main.py --pc=0.8 --pm=0.2 --population=${population[$i]}' ENTER" ENTER
done

PC_WINDOW=("pc_2" "pc_3" "pc_4" "pc_5" "pc_6" "pc_7" "pc_8")
index=(0 1 2 3 4 5 6)
pc=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux new-window -n ${PC_WINDOW[$i]} " ENTER
done

for i in "${index[@]}"
do
    tmux send-keys -t real "tmux send-keys -t ${PC_WINDOW[$i]} 'python main.py --pc=${pc[$i]} --pm=0.2 --population=30' ENTER" ENTER
done

tmux send-keys -t real "tmux new-window -n best_only_false " ENTER
tmux send-keys -t real "tmux new-window -n best_only_true " ENTER
sleep 1

tmux send-keys -t real "tmux send-keys -t best_only_false 'python main.py --pc=0.8 --pm=0.2 --population=30 --select_best_only=False' ENTER" ENTER
tmux send-keys -t real "tmux send-keys -t best_only_true 'python main.py --pc=0.8 --pm=0.2 --population=30 --select_best_only=True' ENTER" ENTER

# End of outer loop.
exit 0