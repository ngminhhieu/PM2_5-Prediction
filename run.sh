#!/bin/bash

if [ ! -d "PM2_5-Prediction" ]
then
git clone -b ga https://github.com/ngminhhieu/PM2_5-Prediction
while [ ! -d "PM2_5-Prediction" ]
do
  sleep 2 # or less like 0.2
done
fi
rm -rf "PM2_5-Prediction/log/*"
cd "PM2_5-Prediction"
tmux new-session -d -s real

WINDOW=( "ffb0" "fffb10" "fffb20" "fffb40" "fffb60" "fffb80" "fffb100")
FIXED_WINDOW=( "ffshb0" "ffshb10" "ffshb20" "ffshb40" "ffshb60" "ffshb80" "ffshb100")
index=(0 1 2 3 4 5 6)
percentage_back_test=(0 10 20 40 60 80 100)
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux new-window -n ${WINDOW[$i]} " ENTER
done
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux new-window -n ${FIXED_WINDOW[$i]} " ENTER
done
sleep 0.5
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux send-keys -t ${WINDOW[$i]} 'python main.py --percentage_back_test=${percentage_back_test[$i]} --fixed=false --tmp=$i' ENTER" ENTER
done
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux send-keys -t ${FIXED_WINDOW[$i]} 'python main.py --percentage_back_test=${percentage_back_test[$i]} --fixed=false --shuffle=false --tmp=$i' ENTER" ENTER
done

# End of outer loop.
exit 0
